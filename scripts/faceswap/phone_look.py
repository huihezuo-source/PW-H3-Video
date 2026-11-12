"""把 H3 重画的人物调成“手机实拍”的质感：皮肤和脸降中频锐度、降明暗反差，整个人略降饱和度（衣服纹理不动），
全画面按亮度加细颗粒（原片手机传感器噪点的水平），缩到 720x1280，音频照搬。
用法：phone_look.py in.mp4 out.mp4 [--detail 0.55] [--contrast 0.7] [--sat 0.82] [--grain 3.0] [--bitrate 1200k]"""
import os, sys, subprocess, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
arg = lambda f, d: float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
src, dst = sys.argv[1], sys.argv[2]
DETAIL, CONTRAST, SAT, GRAIN = arg("--detail", 0.55), arg("--contrast", 0.6), arg("--sat", 0.82), arg("--grain", 3.0)
W, H = 720, 1280
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=os.path.expanduser("~/models/face/selfie_multiclass.tflite")), output_category_mask=True))
cap = cv2.VideoCapture(src); fps = cap.get(5)
tmp = dst + ".video.mp4"
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
rng = np.random.default_rng(0)
while True:
    ok, img = cap.read()
    if not ok: break
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).category_mask.numpy_view()
    soft = lambda x: cv2.GaussianBlur(cv2.dilate(x.astype(np.float32), np.ones((7, 7), np.uint8)), (0, 0), 4)[..., None]
    m = soft(np.isin(cm, [1, 2, 3, 4]))   # 整个人：只降饱和度
    ms = soft(np.isin(cm, [2, 3]))        # 皮肤和脸：降锐度、降反差（衣服纹理保留）
    f = img.astype(np.float32)
    b1, b3 = cv2.GaussianBlur(f, (0, 0), 1.0), cv2.GaussianBlur(f, (0, 0), 3.0)
    p = f - (1 - DETAIL) * (b1 - b3)                     # 中频（发脆的轮廓和纹理）减弱
    big = cv2.GaussianBlur(p, (0, 0), 25)
    p = big + (p - big) * CONTRAST                       # 局部明暗反差减弱（光影没那么硬）
    p = ms * p + (1 - ms) * f
    ycc = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc[..., 1:] = 128 + (ycc[..., 1:] - 128) * SAT       # 饱和度
    p = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
    out = m * p + (1 - m) * f
    luma = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.6) * 1.9  # 略带空间相关，像传感器颗粒
    out += (n * GRAIN * np.sqrt(np.clip(luma, 8, 255) / 190.0))[..., None]
    ff.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
ff.stdin.close(); ff.wait()
# 最后按手机原片的码率压一遍（原片约 1.1 Mbps），让编码器像手机一样把颗粒和皮肤细节抹平
BR = sys.argv[sys.argv.index("--bitrate") + 1] if "--bitrate" in sys.argv else "1200k"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-b:v", BR, "-maxrate", "1800k", "-bufsize", "2400k",
                "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", dst], check=True)
os.remove(tmp)
print(dst)
