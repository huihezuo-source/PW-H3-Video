"""逐句口型检查：按 lines.json 每 0.25 秒截两个角色的头部特写，标出这时该谁说话。
用法：speaker_strip.py video.mp4 lines.json out.png "左角色名:x0,y0,x1,y1" "右角色名:x0,y0,x1,y1" [--from 0 --to 8]（坐标按 576x1024）"""
import sys, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
v, lj, out = sys.argv[1], sys.argv[2], sys.argv[3]
boxes = [(a.split(":")[0], [int(x) for x in a.split(":")[1].split(",")]) for a in sys.argv[4:6]]
arg = lambda f, d: float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
lines = json.load(open(lj))["lines"]; font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 20)
cap = cv2.VideoCapture(v); cols = []
for t in np.arange(arg("--from", 0.25), arg("--to", 8.0), 0.25):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * 24))); ok, img = cap.read()
    if not ok: break
    img = cv2.resize(img, (576, 1024))
    who = "、".join(l["who"] for l in lines if l["start"] <= t <= l["end"]) or "—"
    tiles = []
    for name, (x0, y0, x1, y1) in boxes:
        c = cv2.resize(img[y0:y1, x0:x1], (150, 150)); tiles.append(c)
    col = Image.fromarray(cv2.cvtColor(np.vstack(tiles), cv2.COLOR_BGR2RGB)); top = Image.new("RGB", (150, 52), "black")
    d = ImageDraw.Draw(top); d.text((4, 2), f"{t:.2f}s", fill=(255, 255, 0), font=font); d.text((4, 26), who[:6], fill=(0, 255, 0), font=font)
    c2 = Image.new("RGB", (150, 352)); c2.paste(top, (0, 0)); c2.paste(col, (0, 52)); cols.append(c2)
rows = [cols[i:i + 16] for i in range(0, len(cols), 16)]
S = Image.new("RGB", (150 * 16, 352 * len(rows)), "white")
for r, row in enumerate(rows):
    for k, c in enumerate(row): S.paste(c, (k * 150, r * 352))
S.save(out); print(out, len(cols))
