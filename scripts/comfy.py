"""ComfyUI 小工具：提交工作流、上传文件、等待并下载输出。其他脚本都 import 它。

环境变量：
  COMFY_URL        ComfyUI 地址，默认 http://127.0.0.1:8188
  COMFY_CLIENT_ID  提交用的 client_id，默认 h3-video。看进度时必须用同一个。
"""
import json, os, subprocess, time, urllib.parse, urllib.request

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
CLIENT_ID = os.environ.get("COMFY_CLIENT_ID", "h3-video")


def _save_ui(ui, prompt):
    """把节点图存进 ComfyUI 左侧「工作流」列表的 Claude任务/ 文件夹，名字取输出前缀。"""
    pre = next((n["inputs"].get("filename_prefix") for n in prompt.values()
                if isinstance(n.get("inputs", {}).get("filename_prefix"), str)), None) or time.strftime("%m%d_%H%M%S")
    name = urllib.parse.quote(f"workflows/Claude任务/{pre.replace('/', '_')}.json", safe="")
    body = json.dumps(ui, ensure_ascii=False).encode()
    urllib.request.urlopen(urllib.request.Request(f"{URL}/userdata/{name}?overwrite=true&full_info=true", body,
                                                  {"Content-Type": "application/json"}, method="POST"), timeout=15)


def submit(prompt):
    """提交 API 格式的工作流，返回 prompt_id。同时附上画布能打开的节点图（见 comfy_ui.py），
    这样 ComfyUI 网页里能看到每个任务的完整工作流，左侧「工作流 → Claude任务」里也会多一份。"""
    data = {"prompt": prompt, "client_id": CLIENT_ID}
    try:
        import comfy_ui
        ui = comfy_ui.api_to_ui(prompt, URL)
        data["extra_data"] = {"extra_pnginfo": {"workflow": ui}}
        _save_ui(ui, prompt)
    except Exception as e:
        print("（节点图没生成，不影响任务：", e, "）")
    body = json.dumps(data).encode()
    req = urllib.request.Request(URL + "/prompt", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    if r.get("node_errors"):
        raise SystemExit(f"工作流有错：{r['node_errors']}")
    return r["prompt_id"]


def free():
    """卸载已缓存的模型并释放内存。换模型或连续跑大模型前调用，可避开重复打补丁时的崩溃。"""
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    urllib.request.urlopen(urllib.request.Request(URL + "/free", body, {"Content-Type": "application/json"}))


def upload(path, name, kind="input"):
    """把本地图片或音频传到 ComfyUI 的 input 目录，名字为 name。"""
    subprocess.run(["curl", "-s", "-m", "120", "-F", f"image=@{path};filename={name}",
                    "-F", "overwrite=true", "-F", f"type={kind}", URL + "/upload/image"],
                   check=True, capture_output=True)
    return name


def wait(pid, timeout=3600, every=8):
    """轮询 /history 直到任务完成或报错，返回 history 记录。"""
    t = time.time()
    while time.time() - t < timeout:
        try:
            h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}", timeout=15))
            if pid in h:
                st = h[pid]["status"]
                if st.get("completed") or st.get("status_str") == "error":
                    return h[pid]
        except Exception:
            pass
        time.sleep(every)
    raise TimeoutError(f"{pid} 超过 {timeout} 秒没完成")


def queued(pid):
    """任务是否还在运行或排队。"""
    q = json.load(urllib.request.urlopen(URL + "/queue", timeout=15))
    return any(x[1] == pid for x in q.get("queue_running", []) + q.get("queue_pending", []))


def alive(timeout=20):
    """ComfyUI 是否在线。跑大模型时接口会慢，超时给长一点。"""
    try:
        urllib.request.urlopen(URL + "/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


def down(checks=4, every=15):
    """连续 checks 次（间隔 every 秒）都连不上才算真断了。只查一次会把“忙得慢”误判成崩溃。"""
    for _ in range(checks):
        if alive():
            return False
        time.sleep(every)
    return True


def run(prompt, out, timeout=3600, retries=3):
    """提交并等待完成，再下载输出。ComfyUI 中途崩溃（配合自动重启的启动脚本）时，
    等它恢复后重新提交，最多 retries 次。返回下载的文件名列表。"""
    for attempt in range(retries + 1):
        while not alive():
            time.sleep(10)
        pid = submit(prompt)
        print("prompt_id", pid, f"(第 {attempt + 1} 次)" if attempt else "", flush=True)
        t, missing = time.time(), 0
        while time.time() - t < timeout:
            if not alive() and down():
                while not alive():
                    time.sleep(10)
                try:   # 恢复后先看任务还在不在：还在说明没重启，只是忙，继续等，不重复提交
                    still = queued(pid) or pid in json.load(urllib.request.urlopen(f"{URL}/history/{pid}", timeout=20))
                except Exception:
                    still = False
                if still:
                    print("ComfyUI 刚才没响应，任务还在，继续等", flush=True)
                    continue
                print("ComfyUI 断开过，任务丢了，重新提交", flush=True)
                break
            try:
                h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}", timeout=15))
                if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") == "error"):
                    return download(h[pid], out)
                # 两次轮询之间崩溃又重启完时 alive() 看不出来：任务既不在历史也不在队列里，连续两次就算丢了
                missing = 0 if pid in h or queued(pid) else missing + 1
                if missing >= 2:
                    print("任务不见了（ComfyUI 可能重启过），重新提交", flush=True)
                    break
            except Exception:
                pass
            time.sleep(8)
        else:
            raise TimeoutError(f"{pid} 超过 {timeout} 秒没完成")
        time.sleep(20)
    raise RuntimeError(f"重试 {retries} 次仍然失败")


def download(rec, out):
    """下载一条 history 记录里的全部图片、音频、视频输出，返回文件名列表。"""
    os.makedirs(out, exist_ok=True)
    for m in rec["status"].get("messages", []):
        if m[0] == "execution_error":
            print("出错节点", m[1].get("node_type"), str(m[1].get("exception_message"))[:400])
    got = []
    for o in rec.get("outputs", {}).values():
        for x in o.get("images", []) + o.get("audio", []) + o.get("video", []) + o.get("gifs", []):
            if x.get("type") != "output":
                continue
            q = urllib.parse.urlencode({"filename": x["filename"], "subfolder": x["subfolder"], "type": "output"})
            urllib.request.urlretrieve(f"{URL}/view?{q}", os.path.join(out, x["filename"]))
            got.append(x["filename"])
    return got
