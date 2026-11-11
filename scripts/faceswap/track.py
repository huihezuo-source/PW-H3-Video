"""逐帧找脸（OpenCV YuNet，模型 ~/models/face/yunet.onnx），没检测到的帧用前后插值，再平滑。输出每帧脸框 [cx, cy, size]（相对宽度）。
用法：python track.py 视频.mp4 out.json"""
import json, sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1])
import os
det = None
rows, W, H = [], None, None
while True:
    ok, f = cap.read()
    if not ok: break
    H, W = f.shape[:2]
    if det is None:
        det = cv2.FaceDetectorYN.create(os.path.expanduser("~/models/face/yunet.onnx"), "", (W, H), 0.6, 0.3, 5)
    _, faces = det.detect(f)
    if faces is not None and len(faces):
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])[:4]
        rows.append([(x + w / 2) / W, (y + h / 2) / H, max(w, h) / W])
    else:
        rows.append(None)
n = len(rows); got = [i for i, r in enumerate(rows) if r]
a = np.array([rows[i] for i in got])
full = np.stack([np.interp(range(n), got, a[:, k]) for k in range(3)], 1)
k = 5
sm = np.array([np.median(full[max(0, i - k):i + k + 1], axis=0) for i in range(n)])
json.dump({"w": W, "h": H, "boxes": sm.tolist()}, open(sys.argv[2], "w"))
print(f"frames {n}, detected {len(got)}; cx {sm[:,0].min():.2f}-{sm[:,0].max():.2f}, cy {sm[:,1].min():.2f}-{sm[:,1].max():.2f}, size {sm[:,2].min():.2f}-{sm[:,2].max():.2f}")
