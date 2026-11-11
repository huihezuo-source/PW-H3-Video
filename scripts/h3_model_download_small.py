"""下载 H3 压缩版补充文件（nvfp4 文本编码器 15.7GB + ref2v turbo 4 步 LoRA 2GB）：每个文件 8 路分段并行、断点续传、下完校验 sha256。
只用 Python 标准库。中途断网或重启后再运行一次，会接着已下的部分继续。

用法：python h3_model_download_small.py [models目录] [日志路径]
先跑 h3_model_download.py 装基础模型；本脚本的 nvfp4 编码器是 32GB 内存机器的默认编码器（h3_ref2va.py 默认用它，H3_CLIP 环境变量可改回 int8）。   （模型放到 ROOT 下对应的 ComfyUI models 子目录，进度写到 LOG）
"""
import hashlib, os, threading, time, urllib.request

BASE = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\ComfyUI\models"
LOG = sys.argv[2] if len(sys.argv) > 2 else r"C:\h3\h3dl_small_log.txt"
N = 8
FILES = [
    ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15687142551, "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
    ("loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 1956193000, "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"),
]


def log(msg):
    s = time.strftime("%m-%d %H:%M:%S ") + msg
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def dst_of(rel):
    return os.path.join(ROOT, *rel.split("/"))


def segment(url, part, start, end):
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": "h3dl"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            log(f"重试 {os.path.basename(part)}: {str(e)[:80]}")
            time.sleep(10)


def fetch(rel, size, sha):
    dst = dst_of(rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        log(f"已存在 {rel}")
        return
    step = size // N + 1
    parts, threads = [], []
    for i in range(N):
        s, e = i * step, min(size - 1, (i + 1) * step - 1)
        if s > e:
            break
        p = f"{dst}.part{i}"
        parts.append(p)
        t = threading.Thread(target=segment, args=(BASE + rel, p, s, e), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    log(f"合并并校验 {rel}")
    h = hashlib.sha256()
    with open(dst + ".tmp", "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
                    h.update(b)
    if h.hexdigest() != sha:
        log(f"校验失败，删掉重下 {rel}")
        os.remove(dst + ".tmp")
        for p in parts:
            os.remove(p)
        return fetch(rel, size, sha)
    os.replace(dst + ".tmp", dst)
    for p in parts:
        os.remove(p)
    log(f"完成，校验通过 {rel}")


def on_disk():
    n = 0
    for rel, size, _ in FILES:
        d = dst_of(rel)
        if os.path.exists(d):
            n += os.path.getsize(d)
            continue
        for i in range(N):
            p = f"{d}.part{i}"
            if os.path.exists(p):
                n += os.path.getsize(p)
    return n


def monitor():
    total = sum(f[1] for f in FILES)
    last, lt = on_disk(), time.time()
    while True:
        time.sleep(120)
        now, t = on_disk(), time.time()
        sp = (now - last) / (t - lt)
        last, lt = now, t
        eta = (total - now) / sp / 3600 if sp > 0 else 0
        log(f"进度 {now / 2**30:.1f}/{total / 2**30:.1f} GB，{sp / 2**20:.2f} MB/s，还要约 {eta:.1f} 小时")


log("开始下载 H3 压缩版（nvfp4 编码器 + ref2v turbo LoRA，共 16 GB）")
threading.Thread(target=monitor, daemon=True).start()
for f in FILES:
    fetch(*f)
log("全部完成")
