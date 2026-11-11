"""去掉原片上固定不动的文字：多帧平均后边缘清楚、且各帧几乎不变的像素 = 文字。逐帧用 OpenCV inpaint 修补。
用法：python deoverlay.py in.mp4 out.mp4   区域写死在下面（720x1280 竖屏）。"""
import os, sys, subprocess, cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
_seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=mpt.BaseOptions(model_asset_path=os.path.expanduser("~/models/face/selfie_multiclass.tflite")), output_category_mask=True))
import json
TRACK = np.array(json.load(open(sys.argv[3]))["boxes"]) if len(sys.argv) > 3 else None
CUR = [0]
def person(img):
    cm = np.ascontiguousarray(_seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).category_mask.numpy_view()).reshape(img.shape[:2])
    m = np.isin(cm, (1, 2, 3, 4, 5)).astype(np.uint8)   # 背景在整帧上可能标成 255，只认 1–5 类
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:                                   # 只要最大的一块（真正的人），文字碎块不算
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    if TRACK is not None:                       # 帽顶以上一律不算人（分割模型会把帽子上方的天空误判成配饰）
        cy, fs = TRACK[CUR[0]][1] * img.shape[0], TRACK[CUR[0]][2] * img.shape[1]
        m[:max(0, int(cy - fs * 0.85))] = 0
    return cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0
def _basis(xx, yy):
    x, y = xx / 720.0, yy / 1280.0
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
def sky_fill(img, rect, y0, y1, x0, x1):
    """标题区是天空：拟合天空的平滑渐变；标题框里和天空差别明显、又不是人的像素就是字，用拟合值填掉。"""
    per = person(img)
    H_, W_ = img.shape[:2]
    band = np.zeros((H_, W_), bool); band[max(0, y0):y1, max(0, x0):x1] = True
    known = band & ~rect & ~per
    yy, xx = np.nonzero(known)
    if len(yy) < 500: return img
    sel = np.random.default_rng(0).choice(len(yy), min(8000, len(yy)), replace=False)
    A = _basis(xx[sel].astype(float), yy[sel].astype(float))
    ry, rx = np.nonzero(rect & ~per)
    B = _basis(rx.astype(float), ry.astype(float))
    fit = np.zeros((len(ry), 3))
    for ch in range(3):
        coef, *_ = np.linalg.lstsq(A, img[yy[sel], xx[sel], ch].astype(float), rcond=None)
        fit[:, ch] = B @ coef
    diff = np.abs(img[ry, rx].astype(float) - fit).max(1)
    txt = np.zeros((H_, W_), np.uint8); txt[ry[diff > 14], rx[diff > 14]] = 1
    txt = (cv2.dilate(txt, np.ones((5, 5), np.uint8)) > 0) & rect & ~per
    out = img.copy()
    full = np.zeros((H_, W_, 3)); full[ry, rx] = fit
    ty, tx = np.nonzero(txt)
    out[ty, tx] = np.clip(full[ty, tx] + np.random.default_rng(1).normal(0, 1.0, (len(ty), 3)), 0, 255).astype(np.uint8)
    return out
src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
fr = np.array(fr); n, H, W = fr.shape[:3]
REGIONS = [((170, 390, 30, 690), range(0, n)),        # 标题三行（y0,y1,x0,x1）全程
           ((410, 575, 110, 370), range(0, 72))]      # 日期“11月5号 早盘”前 3 秒
masks = []
for (y0, y1, x0, x1), span in REGIONS:
    g = fr[list(span), y0:y1, x0:x1].astype(np.float32).mean(-1)
    med = np.median(g, axis=0)
    static = (np.abs(g - med) < 10).mean(0) > 0.7          # 七成以上的帧不变就算文字（人头偶尔挡住也不影响）
    edge = np.abs(med - cv2.GaussianBlur(med, (0, 0), 6))
    m = ((edge > 6) & static).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
    full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = m
    masks.append((full, set(span)))
    cv2.imwrite(out + f".mask{len(masks)}.png", full)
_ys, _xs = np.nonzero(masks[0][0])
TR = (max(0, _ys.min() - 14), min(H, _ys.max() + 14), max(0, _xs.min() - 14), min(W, _xs.max() + 14))
TITLE_RECT = np.zeros((H, W), bool); TITLE_RECT[TR[0]:TR[1], TR[2]:TR[3]] = True
print("title rect", TR)
# 日期字幕区：从字消失后的干净帧借背景。只用地平线一条（房子、树、田埂）估计相似变换，校验不过就退回普通修补
REF = 72
orb = cv2.ORB_create(4000)
feat = np.zeros((H, W), np.uint8); feat[380:640, :] = 255
feat[int(H * 0.40):, int(W * 0.22):int(W * 0.95)] = 0       # 人物
for mk, _ in masks: feat[cv2.dilate(mk, np.ones((15, 15), np.uint8)) > 0] = 0
kr, dr = orb.detectAndCompute(cv2.cvtColor(fr[REF], cv2.COLOR_BGR2GRAY), feat)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
date_mask, date_span = masks[1]
dm = cv2.dilate(date_mask, np.ones((9, 9), np.uint8))
soft = cv2.GaussianBlur(dm.astype(np.float32) / 255, (0, 0), 4)[..., None]
log = []
vw = cv2.VideoWriter(out + ".raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
for i in range(n):
    CUR[0] = i
    f = fr[i].copy(); used = False
    if i in date_span:
        ki, di = orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), feat)
        if di is not None and len(ki) > 30:
            ms = sorted(bf.match(dr, di), key=lambda m: m.distance)[:300]
            A = np.float32([kr[m.queryIdx].pt for m in ms]); B = np.float32([ki[m.trainIdx].pt for m in ms])
            M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if M is not None:
                sc = np.hypot(M[0, 0], M[1, 0]); ni = int(inl.sum())
                if ni >= 25 and 0.9 < sc < 1.1:
                    warped = cv2.warpAffine(fr[REF], M, (W, H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                    # 亮度校正：遮罩外一圈的颜色差，按上下两半分别补（天空和地面曝光不同）
                    ring = (cv2.dilate(dm, np.ones((31, 31), np.uint8)) > 0) & (dm == 0)
                    fy = f.astype(np.float32)
                    for ys in (slice(0, 480), slice(480, H)):
                        r = ring.copy(); r[:ys.start or 0] = False; r[ys.stop:] = False
                        if r.sum() > 200:
                            warped[ys] += (fy[r] - warped[r]).mean(0)
                    f = (f.astype(np.float32) * (1 - soft) + warped * soft).clip(0, 255).astype(np.uint8); used = True
                log.append((i, ni, round(float(sc), 3)))
        if not used:
            f = cv2.inpaint(f, date_mask, 6, cv2.INPAINT_TELEA)
    vw.write(sky_fill(f, TITLE_RECT, TR[0] - 60, TR[1] + 30, 0, W))
print("date frames borrowed:", sum(1 for x in log if x[1] >= 25 and 0.9 < x[2] < 1.1), "of", len(date_span), "sample", log[::10])
vw.release()
subprocess.run(["ffmpeg","-v","error","-y","-i",out + ".raw.mp4","-c:v","libx264","-crf","12","-pix_fmt","yuv420p","-r","24",out], check=True)
import os; os.remove(out + ".raw.mp4"); print("done", n)
