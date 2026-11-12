"""量眼球的细微转动：MediaPipe FaceLandmarker 的虹膜点（468-477）相对眼角、眼睑的位置。
输出：水平/垂直视线的波动（std）、每帧平均移动量、扫视次数（一帧内跳动超过眼宽 3%）、长时间不动的最长时长。
头部转动时眼球相对眼眶也会动，这是正常的。用法：eye_motion.py 视频..."""
import os, sys, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
R_EYE, L_EYE = (33, 133, 159, 145, 468), (362, 263, 386, 374, 473)   # 外角、内角、上睑、下睑、虹膜中心
def gaze(pts, e):
    a, b, up, lo, iris = (pts[i] for i in e)
    w = np.linalg.norm(b - a); axis = (b - a) / w
    x = np.dot(iris - a, axis) / w                 # 0..1 沿眼裂方向
    y = (iris[1] - (up[1] + lo[1]) / 2) / w        # 相对眼睛中线，按眼宽归一
    return x, y
def run(path):
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.expanduser("~/models/face/face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    cap = cv2.VideoCapture(path); fps = cap.get(5); i = 0; g = []
    while True:
        ok, img = cap.read()
        if not ok or i >= 340: break
        h, w = img.shape[:2]
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps)); i += 1
        if not r.face_landmarks: g.append(None); continue
        blink = (r.face_blendshapes[0][9].score + r.face_blendshapes[0][10].score) / 2  # eyeBlinkLeft/Right
        pts = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
        (xr, yr), (xl, yl) = gaze(pts, R_EYE), gaze(pts, L_EYE)
        g.append(None if blink > 0.35 else ((xr + xl) / 2, (yr + yl) / 2))
    v = np.array([x for x in g if x is not None])
    d = []; still = run_len = 0
    for a, b in zip(g, g[1:]):
        if a is None or b is None: run_len = 0; continue
        dd = np.hypot(b[0] - a[0], b[1] - a[1]); d.append(dd)
        run_len = run_len + 1 if dd < 0.006 else 0; still = max(still, run_len)
    d = np.array(d)
    return dict(gaze_x_std=v[:, 0].std(), gaze_y_std=v[:, 1].std(), move_per_frame=d.mean(), saccades=int((d > 0.03).sum()), longest_still_s=still / fps)
for p in sys.argv[1:]:
    r = run(p)
    print(p.split("/")[-1][:26].ljust(26), " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
