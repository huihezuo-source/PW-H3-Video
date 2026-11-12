"""给 AI 视频里的眼球加真人那样的细微跳动（微扫视）：H3 在口播尺度下眼睛只有十几像素，画不出这种细节。
逐帧用 MediaPipe FaceLandmarker 找虹膜和眼眶，按“注视 0.25–0.7 秒 → 1–2 帧内快速跳到附近一点”的节奏生成偏移，
两眼同步；只在眼眶内、以虹膜为中心局部平移（眼皮不动），眨眼时不加。
用法：eye_saccades.py in.mp4 out.mp4 [--amp-x 0.045] [--amp-y 0.02] [--seed 1]（幅度单位：眼宽）"""
import os, sys, subprocess, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
arg = lambda f, d: float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
src, dst = sys.argv[1], sys.argv[2]
AX, AY, SEED = arg("--amp-x", 0.045), arg("--amp-y", 0.02), int(arg("--seed", 1))
EYES = {  # 眼眶轮廓、外角、内角、虹膜中心、虹膜边缘
    "r": ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], 33, 133, 468, [469, 470, 471, 472]),
    "l": ([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], 362, 263, 473, [474, 475, 476, 477]),
}
cap = cv2.VideoCapture(src); fps = cap.get(5); n = int(cap.get(7)); W, H = int(cap.get(3)), int(cap.get(4))
# 微扫视轨迹：一段段注视，段间 2 帧内平滑跳过去
rng = np.random.default_rng(SEED); traj = np.zeros((n + 2, 2)); t = 0; cur = np.zeros(2)
while t < n:
    hold = int(rng.uniform(0.25, 0.7) * fps)
    ang = rng.uniform(0, 2 * np.pi); rad = np.sqrt(rng.uniform(0.15, 1.0))
    nxt = np.array([np.cos(ang) * AX * rad, np.sin(ang) * AY * rad])
    for k in range(2):                      # 2 帧完成跳动
        if t + k < n: traj[t + k] = cur + (nxt - cur) * [0.6, 1.0][k]
    traj[t + 2:t + hold] = nxt; cur = nxt; t += max(hold, 3)
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.expanduser("~/models/face/face_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
tmp = dst + ".video.mp4"
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
widths = []
for i in range(n):
    ok, img = cap.read()
    if not ok: break
    r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
    if r.face_landmarks:
        bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
        openness = 1 - max(bs["eyeBlinkLeft"], bs["eyeBlinkRight"])
        pts = np.array([[p.x * W, p.y * H] for p in r.face_landmarks[0]], np.float32)
        mapx, mapy = gx.copy(), gy.copy()
        for contour, a, b, ic, edge in EYES.values():
            ew = np.linalg.norm(pts[a] - pts[b]); widths.append(ew)
            c = pts[ic]; ir = np.mean([np.linalg.norm(pts[e] - c) for e in edge])
            d = traj[i] * ew * np.clip((openness - 0.35) / 0.3, 0, 1)   # 眨眼时不动
            m = np.zeros((H, W), np.float32)
            cv2.fillPoly(m, [pts[contour].astype(np.int32)], 1.0)
            m = cv2.GaussianBlur(m, (0, 0), max(1.0, ew * 0.04))
            wgt = m * np.exp(-((gx - c[0]) ** 2 + (gy - c[1]) ** 2) / (2 * (ir * 1.5) ** 2))
            mapx -= d[0] * wgt; mapy -= d[1] * wgt
        img = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    ff.stdin.write(img.tobytes())
ff.stdin.close(); ff.wait()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", src, "-map", "0:v", "-map", "1:a?", "-c", "copy", "-shortest", dst], check=True)
os.remove(tmp)
print(dst, "median eye width px", round(float(np.median(widths)), 1), "max shift px", round(float(np.abs(traj).max(0)[0] * np.median(widths)), 2))
