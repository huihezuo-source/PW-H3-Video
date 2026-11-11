"""逐句说话人检查：按台词表，每句台词的时间段出一张密集帧网格。

看每张网格只回答一个封闭题：谁的嘴在动？A / B / 都动 / 都不动。再和台词表的说话人比对。
旁白句的正确答案是“都不动”。说话人错了不能交付。

用法：python3 speaker_grid.py video.mp4 lines.json out_dir [--step 0.25] [--width 320]
lines.json：[[开始秒, 结束秒, "说话人", "台词"], ...]
依赖：ffmpeg、Pillow
"""
import json, math, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

video, lines, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
arg = lambda f, d: float(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
FONT = None
for p in ("/System/Library/Fonts/STHeiti Medium.ttc", "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(p):
        FONT = ImageFont.truetype(p, 22)
        break
step, width = arg("--step", 0.25), int(arg("--width", 320))
os.makedirs(out, exist_ok=True)

for n, (a, b, who, text) in enumerate(lines, 1):
    times = [a + k * step for k in range(int((b - a) / step) + 1)]
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, t in enumerate(times):
            f = os.path.join(tmp, f"{k}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                            "-vf", f"scale={width}:-2", f], check=True)
            if os.path.exists(f):
                im = Image.open(f).convert("RGB")
                ImageDraw.Draw(im).text((6, 6), f"{t:.2f}s", fill=(255, 255, 0))
                tiles.append(im)
    if not tiles:
        print(f"#{n} {a:.2f}-{b:.2f}s 取不到帧，检查时间是否超出视频长度")
        continue
    cols = min(6, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, rows * h + 40), (20, 20, 20))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * w, 40 + (k // cols) * h))
    ImageDraw.Draw(sheet).text((8, 12), f"#{n} {a:.2f}-{b:.2f}s  应为：{who}  {text}", fill=(255, 255, 255), font=FONT)
    path = os.path.join(out, f"line_{n:02d}.jpg")
    sheet.save(path, quality=88)
    print(path, len(tiles), "帧", "应为", who)
