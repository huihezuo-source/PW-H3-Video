"""沿着人脸跟踪结果裁出一个跟随人脸的窗口（稳定的脸部特写），放大到 H3 尺寸。
用法：python crop.py 底片.mp4 track.json out_crop.mp4 OW OH [窗口宽占底片宽的比例 0.7]
窗口宽高比 = OW:OH；窗口位置写进 out_crop.json 供贴回用。"""
import json, sys, cv2, numpy as np
src, tj, out, OW, OH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
frac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7
t = json.load(open(tj)); boxes = np.array(t["boxes"])
cap = cv2.VideoCapture(src); W = int(cap.get(3)); H = int(cap.get(4))
cw = int(W * frac); ch = int(cw * OH / OW)
# 窗口中心再平滑一次（更稳），脸在窗口里略偏上，给下巴和嘴留空间
k = 12
cx = np.array([boxes[max(0, i-k):i+k+1, 0].mean() for i in range(len(boxes))]) * W
cy = np.array([boxes[max(0, i-k):i+k+1, 1].mean() for i in range(len(boxes))]) * H + ch * 0.05
wins = []
vw = cv2.VideoWriter(out + ".raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (OW, OH))
i = 0
while True:
    ok, f = cap.read()
    if not ok or i >= len(cx): break
    x0 = int(np.clip(cx[i] - cw / 2, 0, W - cw)); y0 = int(np.clip(cy[i] - ch / 2, 0, H - ch))
    wins.append([x0, y0, cw, ch])
    vw.write(cv2.resize(f[y0:y0+ch, x0:x0+cw], (OW, OH), interpolation=cv2.INTER_LANCZOS4))
    i += 1
vw.release()
json.dump({"W": W, "H": H, "OW": OW, "OH": OH, "wins": wins}, open(out.replace(".mp4", ".json"), "w"))
import subprocess
subprocess.run(["ffmpeg","-v","error","-y","-i",out + ".raw.mp4","-c:v","libx264","-crf","12","-pix_fmt","yuv420p","-r","24",out], check=True)
import os; os.remove(out + ".raw.mp4")
print("window", cw, "x", ch, "->", OW, "x", OH, "frames", len(wins))
