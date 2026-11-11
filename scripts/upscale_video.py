"""RealESRGAN 逐帧放大视频，再 lanczos 缩放到目标尺寸，保留原音轨。只放大已经确认的版本。

用法：python3 upscale_video.py in.mp4 out.mp4 RealESRGAN_x2.pth [目标宽 目标高]
依赖：pip install spandrel torch；Mac 用 mps，N 卡用 cuda，否则 cpu。
"""
import json, subprocess, sys, time
import numpy as np, torch
from spandrel import ModelLoader

src, dst, model_path = sys.argv[1:4]
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                                  "-of", "json", src], capture_output=True, text=True).stdout)["streams"][0]
W, H, fps = info["width"], info["height"], info["r_frame_rate"]
OW, OH = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (W * 2, H * 2)
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
m = ModelLoader().load_from_file(model_path).to(dev).eval()
dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * m.scale}x{H * m.scale}", "-r", fps,
                        "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?", "-vf", f"scale={OW}:{OH}:flags=lanczos",
                        "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy",
                        "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
n, t = 0, time.time()
with torch.no_grad():
    while True:
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            break
        x = torch.from_numpy(np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()).permute(2, 0, 1)[None].float().div(255).to(dev)
        enc.stdin.write(m(x).clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy().tobytes())
        n += 1
        if n % 60 == 0:
            print(n, f"{time.time() - t:.0f}s", flush=True)
enc.stdin.close(); enc.wait()
print("done", n, f"{time.time() - t:.0f}s")
