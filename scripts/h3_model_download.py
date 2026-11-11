"""在网速慢的电脑上下载 H3 模型：每个文件 8 路分段并行、断点续传、下完校验 sha256。
只用 Python 标准库。中途断网或重启后再运行一次，会接着已下的部分继续。

用法：python h3_model_download.py   （模型放到 ROOT 下对应的 ComfyUI models 子目录，进度写到 LOG）
"""
import hashlib, os, threading, time, urllib.request

BASE = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
ROOT = r"D:\ComfyUI\models"
LOG = r"D:\h3dl_log.txt"
N = 8
FILES = [
    ("vae/minimax_h3_audio_vae_fp32.safetensors", 605254808, "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
    ("vae/minimax_h3_video_vae_int8_convrot.safetensors", 2811065184, "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e"),
    ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", 20970379616, "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"),
    ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", 27141342152, "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"),
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


log("开始下载 H3 模型（共 48 GB）")
threading.Thread(target=monitor, daemon=True).start()
for f in FILES:
    fetch(*f)
log("全部完成")
