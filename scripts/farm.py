"""多机并行出片：把一批 H3 任务分给几台 ComfyUI 机器同时跑，谁空闲谁接下一条。

用法：python3 farm.py jobs.json out_dir [--machines machines.json] [--script h3_ref2va.py]
jobs.json：[{"name": "seg1", "job": "seg1.json"}, {"name": "seg2", "job": "seg2.json"}, ...]
            每个 job 文件就是 h3_ref2va.py（或 --script 指定的脚本）平时吃的那种任务文件。
machines.json：[{"name": "WJ", "url": "http://127.0.0.1:18188"}, {"name": "XW", "url": "http://127.0.0.1:18189"}]
            不给就用同目录的 farm_machines.json。打不通的机器自动跳过，任务只分给在线的机器。

每条任务在对应机器上跑完后，输出下载到 out_dir/<name>/。一条任务失败会换一台机器重试一次。
进度写到 out_dir/farm_log.txt，同时打印。工作台（dashboard.py）里能看到每台机器的实时步数。
"""
import json, os, queue, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def log(out, msg):
    s = time.strftime("%H:%M:%S ") + msg
    print(s, flush=True)
    with open(os.path.join(out, "farm_log.txt"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


def alive(url):
    try:
        urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=10)
        return True
    except Exception:
        return False


def worker(m, q, out, script, results):
    env = dict(os.environ, COMFY_URL=m["url"], COMFY_CLIENT_ID="h3-video")
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        name = job["name"]
        dst = os.path.join(out, name)
        os.makedirs(dst, exist_ok=True)
        log(out, f"[{m['name']}] 开始 {name}")
        t = time.time()
        logf = open(os.path.join(dst, "run_log.txt"), "a", encoding="utf-8")
        r = subprocess.run([sys.executable, os.path.join(HERE, script), job["job"], "--wait", dst],
                           env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(job["job"])) or ".")
        logf.close()
        mins = (time.time() - t) / 60
        if r.returncode == 0:
            log(out, f"[{m['name']}] 完成 {name}，{mins:.1f} 分钟")
            results[name] = m["name"]
        else:
            job["tries"] = job.get("tries", 0) + 1
            log(out, f"[{m['name']}] 失败 {name}（第 {job['tries']} 次），{mins:.1f} 分钟，详情 {dst}/run_log.txt")
            if job["tries"] < 2:
                q.put(job)
            else:
                results[name] = None
        q.task_done()


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    jobs_file, out = a[0], a[1]
    mfile = a[a.index("--machines") + 1] if "--machines" in a else os.path.join(HERE, "farm_machines.json")
    script = a[a.index("--script") + 1] if "--script" in a else "h3_ref2va.py"
    os.makedirs(out, exist_ok=True)
    machines = [m for m in json.load(open(mfile)) if alive(m["url"])]
    offline = [m["name"] for m in json.load(open(mfile)) if m["name"] not in [x["name"] for x in machines]]
    if offline:
        log(out, f"离线，跳过：{' '.join(offline)}")
    if not machines:
        raise SystemExit("没有一台机器在线，检查 RustDesk 隧道或 Tailscale")
    jobs = json.load(open(jobs_file))
    base = os.path.dirname(os.path.abspath(jobs_file))
    for j in jobs:
        if not os.path.isabs(j["job"]):
            j["job"] = os.path.join(base, j["job"])
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    log(out, f"{len(jobs)} 条任务，{len(machines)} 台机器：{' '.join(m['name'] for m in machines)}")
    results = {}
    ths = [threading.Thread(target=worker, args=(m, q, out, script, results), daemon=True) for m in machines]
    for t in ths:
        t.start()
        time.sleep(2)
    for t in ths:
        t.join()
    done = [n for n, m in results.items() if m]
    failed = [n for n, m in results.items() if not m]
    log(out, f"全部结束：成功 {len(done)}，失败 {len(failed)}{'：' + ' '.join(failed) if failed else ''}")
    json.dump(results, open(os.path.join(out, "farm_result.json"), "w"), ensure_ascii=False, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
