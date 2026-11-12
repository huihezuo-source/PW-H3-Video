"""量“像不像手机实拍”：人物区域和背景区域各自的噪点、细节锐度、饱和度、亮度，和原片对比。
人物区域用 MediaPipe 多分类分割（头发+皮肤+脸+衣服），背景取其余部分。用法：look_stats.py 视频1 视频2 ..."""
import os, sys, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=os.path.expanduser("~/models/face/selfie_multiclass.tflite")), output_category_mask=True))
def stats(path, n=24):
    cap = cv2.VideoCapture(path); total = int(cap.get(7)); res = {"person": [], "bg": [], "face": []}
    for i in np.linspace(10, total - 10, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, img = cap.read()
        if not ok: continue
        img = cv2.resize(img, (720, 1280), interpolation=cv2.INTER_AREA)
        cm = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).category_mask.numpy_view()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        noise = g - cv2.GaussianBlur(g, (0, 0), 1.0)         # 最细一层：噪点和毛孔
        detail = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 3.0)  # 中频：轮廓和纹理
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        masks = {"person": np.isin(cm, [1, 2, 3, 4]), "face": cm == 3, "bg": cm == 0}
        for k, m in masks.items():
            m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            if m.sum() < 500: continue
            res[k].append([noise[m].std(), detail[m].std(), hsv[..., 1][m].mean(), g[m].mean(), g[m].std()])
    return {k: np.array(v).mean(0) for k, v in res.items() if v}
print(f"{'video':28s} {'area':7s} {'noise':>6s} {'detail':>6s} {'sat':>6s} {'luma':>6s} {'contrast':>8s}")
for p in sys.argv[1:]:
    for k, v in stats(p).items():
        print(f"{p.split('/')[-1][:28]:28s} {k:7s} " + " ".join(f"{x:6.2f}" for x in v))
