"""Windows 渲染机稳定性压力测试：CPU + 内存校验，再连跑几条真实 H3 任务（每条前卸载模型、重新加载）。

用来判断一台机器会不会像家里那台 13900K 一样蓝屏、ComfyUI 崩溃。只用标准库 + ComfyUI venv 里的 numpy。
每个 CPU 进程反复做 zlib 压缩/解压、矩阵乘法、整块内存 sha256，并和第一次的结果逐字比对，
算错一次就记一个 ERROR（不稳定的 CPU/内存最先表现为算错，而不是马上蓝屏）。

用法（在渲染机上，用 ComfyUI 的 venv 跑；ComfyUI 先在 8188 起好）：
  python stress_test.py [--cpu-min 15] [--jobs 3] [--workers 32] [--mb 1100]
日志：./st_test\\log.txt，监控：D:\\stress\\monitor.csv，结论：D:\\stress\\summary.txt
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, hashlib, json, multiprocessing as mp, subprocess, threading, time, urllib.request, zlib

DIR = "./st_test"
URL = "http://127.0.0.1:8188"
PROMPT = ("integrated_multimodal_description: [0.0s-10.1s] Anime style, rainy night street, wide shot. "
          "A young man in a gray hoodie walks toward the camera under pink and blue neon signs, rain falling, "
          "puddles reflecting the neon. The camera slowly dollies back. At 6s he stops, looks up at the sky and exhales. "
          "overall_soundscape: steady rain, distant traffic, footsteps on wet asphalt. non_diegetic_music: none.")


def log(msg):
    s = time.strftime("%m-%d %H:%M:%S ") + msg
    print(s, flush=True)
    with open(os.path.join(DIR, "log.txt"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


def cpu_worker(idx, secs, mb, errq, cntq):
    import numpy as np
    rng = np.random.default_rng(idx)
    buf = rng.integers(0, 256, size=mb * 2**20, dtype=np.uint8)
    ref = hashlib.sha256(buf).hexdigest()
    text = bytes(buf[: 8 * 2**20] & 0x0F)
    zref = len(zlib.compress(text, 6))
    a = rng.standard_normal((384, 384))
    mref = float((a @ a).sum())
    end, n, err = time.time() + secs, 0, 0
    while time.time() < end:
        z = zlib.compress(text, 6)
        if len(z) != zref or zlib.decompress(z) != text:
            err += 1; errq.put(f"worker {idx}: zlib mismatch")
        m = float((a @ a).sum())
        if m != mref:
            err += 1; errq.put(f"worker {idx}: matmul mismatch {m!r} vs {mref!r}")
        if n % 10 == 0 and hashlib.sha256(buf).hexdigest() != ref:
            err += 1; errq.put(f"worker {idx}: memory sha256 mismatch")
        n += 1
    cntq.put((idx, n, err))


def run_cpu(workers, secs, mb, tag):
    errq, cntq = mp.Queue(), mp.Queue()
    ps = [mp.Process(target=cpu_worker, args=(i, secs, mb, errq, cntq), daemon=True) for i in range(workers)]
    for p in ps:
        p.start()
    log(f"{tag}: {workers} workers x {mb} MB for {secs // 60} min")
    return ps, errq, cntq


def finish_cpu(ps, errq, cntq, tag):
    for p in ps:
        p.join()
    errs = []
    while not errq.empty():
        errs.append(errq.get())
    total = 0
    while not cntq.empty():
        total += cntq.get()[1]
    bad = [p.exitcode for p in ps if p.exitcode != 0]
    for e in errs[:20]:
        log("ERROR " + e)
    log(f"{tag} done: {total} loops, {len(errs)} compute errors, {len(bad)} workers died {bad}")
    return len(errs) + len(bad)


def monitor(stop):
    path = os.path.join(DIR, "monitor.csv")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("time,gpu_temp,gpu_w,gpu_util,gpu_mem_mb,gpu_clock,cpu_busy,cpu_perf_pct,ram_free_mb\n")
        while not stop.is_set():
            try:
                g = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
                t = subprocess.run(["typeperf", "-sc", "1", r"\Processor(_Total)\% Processor Time",
                                    r"\Processor Information(_Total)\% Processor Performance", r"\Memory\Available MBytes"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
                c = [x.strip('"') for x in t[2].split('","')][1:] if len(t) > 2 else ["", "", ""]
                f.write(time.strftime("%H:%M:%S,") + ",".join(x.strip() for x in g.split(",")) + ","
                        + ",".join(f"{float(x):.0f}" if x else "" for x in c) + "\n")
                f.flush()
            except Exception as e:
                f.write(time.strftime("%H:%M:%S,") + "monitor error " + str(e)[:60] + "\n")
            stop.wait(10)


def api(path, body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"} if data else {})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def alive():
    try:
        api("/system_stats", timeout=20)
        return True
    except Exception:
        return False


def workflow(seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["3", 0], "prompt": PROMPT, "width": 896, "height": 384,
               "length": 243, "ref_image_size": "match", "vae": ["4", 0], "audio_vae": ["5", 0]}},
        "40": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["30", 0]}},
        "41": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "42": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "44": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["43", 0], "guider": ["40", 0], "sampler": ["41", 0],
               "sigmas": ["42", 0], "latent_image": ["30", 1]}},
        "45": {"class_type": "VAEDecode", "inputs": {"samples": ["44", 0], "vae": ["4", 0]}},
        "46": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["44", 0], "vae": ["5", 0]}},
        "47": {"class_type": "CreateVideo", "inputs": {"images": ["45", 0], "audio": ["46", 0], "fps": 24}},
        "48": {"class_type": "SaveVideo", "inputs": {"video": ["47", 0], "filename_prefix": f"stress/h3_seed{seed}", "format": "mp4", "codec": "h264"}},
    }


def run_job(seed, tries=3):
    for attempt in range(1, tries + 1):
        while not alive():
            log("ComfyUI not responding, waiting")
            time.sleep(15)
        try:
            api("/free", {"unload_models": True, "free_memory": True})
        except Exception:
            pass
        time.sleep(5)
        t0 = time.time()
        r = api("/prompt", {"prompt": workflow(seed), "client_id": "stress"})
        if r.get("node_errors"):
            log(f"job seed {seed}: workflow error {str(r['node_errors'])[:300]}")
            return "workflow_error", 0
        pid = r["prompt_id"]
        log(f"job seed {seed} attempt {attempt} submitted {pid}")
        down = 0
        while True:
            time.sleep(10)
            if not alive():
                down += 1
                if down >= 4:
                    log(f"CRASH: ComfyUI went down during job seed {seed} ({time.time() - t0:.0f}s in)")
                    break
                continue
            down = 0
            try:
                h = api(f"/history/{pid}")
            except Exception:
                continue
            if pid in h:
                st = h[pid]["status"]
                if st.get("status_str") == "error":
                    msg = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                    log(f"job seed {seed} ERROR: {str(msg)[:400]}")
                    return "error", time.time() - t0
                if st.get("completed"):
                    dt = time.time() - t0
                    log(f"job seed {seed} OK in {dt / 60:.1f} min")
                    return "ok", dt
    return "crashed", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-min", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--mb", type=int, default=1100)
    a = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    log("===== stress test start =====")
    problems = 0
    if a.cpu_min > 0:
        ps, eq, cq = run_cpu(a.workers, 6, a.mb, "phase1 CPU+RAM")
        problems += finish_cpu(ps, eq, cq, "phase1 CPU+RAM")
    results = []
    for i in range(0):
        bg = run_cpu(16, 12 * 60, 64, "phase2 background CPU load") if i == 0 else None
        results.append(run_job(100 + i))
        if bg:
            problems += finish_cpu(*bg, "phase2 background CPU load")
    stop.set()
    ok = sum(1 for r in results if r[0] == "ok")
    summary = (f"compute/worker errors: {problems}; H3 jobs ok {ok}/{len(results)}; "
               f"results {[(r[0], round(r[1] / 60, 1)) for r in results]}")
    log("===== SUMMARY " + summary)
    with open(os.path.join(DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
