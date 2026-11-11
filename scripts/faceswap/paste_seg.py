"""按人像分割贴回：头（头发、脸、帽子、眼镜）+ 脖子用重画的，衣服和背景用底片的，边缘只羽化几个像素。
分割模型 ~/models/face/selfie_multiclass.tflite（0 背景 1 头发 2 身体皮肤 3 脸 4 衣服 5 配饰）。
用法：python paste_seg.py 底片.mp4 重画窗口.mp4 crop.json track.json out.mp4"""
import json, os, sys, subprocess, cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
base, fix, cj, tj, out = sys.argv[1:6]
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=mpt.BaseOptions(model_asset_path=os.path.expanduser("~/models/face/selfie_multiclass.tflite")), output_category_mask=True))
c = json.load(open(cj)); t = np.array(json.load(open(tj))["boxes"]); W, H = c["W"], c["H"]
det = {}
def landmarks(img):
    """YuNet 五点：右眼、左眼、鼻尖、右嘴角、左嘴角。"""
    k = img.shape[:2]
    if k not in det:
        det[k] = cv2.FaceDetectorYN.create(os.path.expanduser("~/models/face/yunet.onnx"), "", (k[1], k[0]), 0.5, 0.3, 5)
    _, faces = det[k].detect(img)
    if faces is None or not len(faces): return None
    f = max(faces, key=lambda b: b[2] * b[3])
    return np.float32(f[4:14]).reshape(5, 2)
def head(img, fy, fs):
    """返回 (core, full)：core = 头发、脸、脖子；full = core 加上头上的配饰（帽子、眼镜）。"""
    cm = np.ascontiguousarray(seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).category_mask.numpy_view()).reshape(img.shape[:2])
    yy = np.arange(img.shape[0])[:, None]
    core = np.isin(cm, (1, 3)) | ((cm == 2) & (yy < fy + fs * 0.85))
    full = core | ((cm == 5) & (yy < fy + fs * 0.95) & (yy > fy - fs * 0.85))
    def biggest(m):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else m
    return biggest(core), biggest(full)
cb, cf = cv2.VideoCapture(base), cv2.VideoCapture(fix)
frames_b, patches_raw, wins, Ms = [], [], [], []
i = 0
while True:
    ok1, fb = cb.read(); ok2, ff = cf.read()
    if not ok1 or not ok2 or i >= len(c["wins"]): break
    x0, y0, cw, ch = c["wins"][i]
    patch = cv2.resize(ff, (cw, ch), interpolation=cv2.INTER_AREA)
    roi = fb[y0:y0+ch, x0:x0+cw]
    lp, lb = landmarks(patch), landmarks(roi)
    M = None
    if lp is not None and lb is not None:
        M, _ = cv2.estimateAffinePartial2D(lp, lb)
    Ms.append(M); frames_b.append(fb); patches_raw.append(patch); wins.append((x0, y0, cw, ch, t[i][1] * H - y0, t[i][2] * W)); i += 1
n = len(frames_b)
# 对齐参数（平移、缩放、角度）缺的帧插值，再 ±4 帧平滑
P = np.full((n, 4), np.nan)
for i, M in enumerate(Ms):
    if M is not None:
        sc = np.hypot(M[0, 0], M[1, 0]); P[i] = (M[0, 2], M[1, 2], sc, np.arctan2(M[1, 0], M[0, 0]))
ok = ~np.isnan(P[:, 0])
for k in range(4):
    P[:, k] = np.interp(np.arange(n), np.where(ok)[0], P[ok, k])
P = np.array([P[max(0, i-4):i+5].mean(0) for i in range(n)])
print("aligned frames", int(ok.sum()), "of", n, "| mean shift %.1f,%.1f px, scale %.3f" % (P[:, 0].mean(), P[:, 1].mean(), P[:, 2].mean()))
patches, masks = [], []
for i in range(n):
    x0, y0, cw, ch, fy, fs = wins[i]
    tx, ty, sc, an = P[i]
    M = np.float32([[sc * np.cos(an), -sc * np.sin(an), tx], [sc * np.sin(an), sc * np.cos(an), ty]])
    patch = cv2.warpAffine(patches_raw[i], M, (cw, ch), borderMode=cv2.BORDER_REPLICATE)
    roi = frames_b[i][y0:y0+ch, x0:x0+cw]
    (cp, fp), (cb_, fb_) = head(patch, fy, fs), head(roi, fy, fs)
    core = cv2.dilate((cp | cb_).astype(np.uint8), np.ones((5, 5), np.uint8))
    extra = cv2.dilate(((fp | fb_) & ~(cp | cb_)).astype(np.uint8), np.ones((5, 5), np.uint8))
    masks.append((cv2.GaussianBlur(core.astype(np.float32), (0, 0), 2.5), extra.astype(np.float32)))
    patches.append(patch.astype(np.float32))
vw = cv2.VideoWriter(out + ".raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
for i in range(n):
    x0, y0, cw, ch, fy, fs = wins[i]
    core = np.mean([masks[j][0] for j in range(max(0, i-1), min(n, i+2))], axis=0)   # 前后帧平滑，边缘不抖
    extra = np.mean([masks[j][1] for j in range(max(0, i-1), min(n, i+2))], axis=0)
    m = np.clip(np.maximum(core, cv2.GaussianBlur(extra, (0, 0), 2.5)), 0, 1)       # 新旧两个头的并集全部用重画的，旧帽子不留
    fb = frames_b[i]; roi = fb[y0:y0+ch, x0:x0+cw].astype(np.float32); patch = patches[i].copy()
    # 局部校色：两边都是背景的像素算逐点色差，用归一化高斯模糊平滑地延伸进头部周围
    bg = (cv2.dilate((m > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0).astype(np.float32)
    wsum = cv2.GaussianBlur(bg, (0, 0), 28) + 1e-4
    corr = np.stack([cv2.GaussianBlur((roi[..., k] - patch[..., k]) * bg, (0, 0), 28) / wsum for k in range(3)], -1)
    corr = np.clip(corr, -30, 30)
    # 脸本身（core 往里收一圈）不吃局部色差，只做整体轻微校色，避免把田地的绿色带到脸上
    face_w = cv2.GaussianBlur(cv2.erode((core > 0.5).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(np.float32), (0, 0), 6)[..., None]
    glob = np.array([((roi[..., k] - patch[..., k]) * bg).sum() / (bg.sum() + 1e-4) for k in range(3)]) * 0.5
    patch += corr * (1 - face_w) + glob * face_w
    m3 = m[..., None]
    fb[y0:y0+ch, x0:x0+cw] = np.clip(roi * (1 - m3) + patch * m3, 0, 255).astype(np.uint8)
    vw.write(fb)
vw.release()
subprocess.run(["ffmpeg","-v","error","-y","-i",out + ".raw.mp4","-c:v","libx264","-crf","12","-pix_fmt","yuv420p","-r","24",out], check=True)
os.remove(out + ".raw.mp4"); print("pasted", n, "frames")
