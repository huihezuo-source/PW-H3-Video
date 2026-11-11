"""出封面：3:4（1080×1440）和 9:16（1080×1920）各一版，上方压暗放两行标题，下方红色标签。

用法：python3 make_cover.py 画面.png 输出目录 "第一行" "第二行" "标签" [--y 0.5] [--model RealESRGAN_x2.pth] [--font 字体.ttc]
--y：3:4 裁切的竖向位置（0 顶，1 底）。给了 --model 会先放大一倍。
"""
import sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

src, out, l1, l2, tag = sys.argv[1:6]
arg = lambda f, d=None: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d
yf = float(arg("--y", 0.5))
font_path = arg("--font", "/System/Library/Fonts/STHeiti Medium.ttc")
base = Image.open(src).convert("RGB")
if arg("--model"):
    import numpy as np, torch
    from spandrel import ModelLoader
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    m = ModelLoader().load_from_file(arg("--model")).to(dev).eval()
    x = torch.from_numpy(np.asarray(base).copy()).permute(2, 0, 1)[None].float().div(255).to(dev)
    with torch.no_grad():
        base = Image.fromarray(m(x).clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy())


def font(sz):
    return ImageFont.truetype(font_path, sz)


def fill(W, H, y):
    bw, bh = base.size
    s = max(W / bw, H / bh)
    im = base.resize((round(bw * s), round(bh * s)), Image.LANCZOS)
    x, y = (im.width - W) // 2, int((im.height - H) * y)
    return im.crop((x, y, x + W, y + H))


def shade(im, frac, a=200):
    W, H = im.size
    g = Image.new("L", (1, H))
    for i in range(H):
        g.putpixel((0, i), int(a * max(0, 1 - i / (H * frac))))
    return Image.composite(Image.new("RGB", (W, H)), im, g.resize((W, H)))


def text(im, top, W, H):
    d = ImageDraw.Draw(im)

    def ct(y, t, sz, col):
        f = font(sz)
        w = d.textlength(t, font=f) + 6
        d.text(((W - w) / 2, y), t, font=f, fill=col, stroke_width=int(sz * 0.13), stroke_fill=(0, 0, 0))
        d.text(((W - w) / 2, y), t, font=f, fill=col, stroke_width=3, stroke_fill=col)
        return y + sz * 1.22

    y = ct(top, l1, int(W * 0.12), (255, 255, 255))
    ct(y, l2, int(W * 0.125), (255, 214, 0))
    sz = int(W * 0.066)
    f = font(sz)
    tw = d.textlength(tag, font=f)
    ph = int(sz * 1.7)
    bx, by = (W - tw) / 2 - 36, H * 0.865
    d.rounded_rectangle((bx, by, bx + tw + 72, by + ph), radius=ph // 2, fill=(214, 32, 32), outline=(255, 255, 255), width=4)
    d.text((bx + 36, by + (ph - sz) / 2 - sz * 0.08), tag, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(255, 255, 255))
    return im


text(shade(fill(1080, 1440, yf), 0.4), 1440 * 0.04, 1080, 1440).save(f"{out}/封面_3比4_1080x1440.jpg", quality=95)
W, H = 1080, 1920
bw, bh = base.size
fg = base.resize((W, round(bh * W / bw)), Image.LANCZOS)
bg = fill(W, H, 0).filter(ImageFilter.GaussianBlur(40))
mk = Image.new("L", fg.size, 255)
md = ImageDraw.Draw(mk)
for i in range(160):
    md.line([(0, i), (W, i)], fill=int(255 * i / 160))
bg.paste(fg, (0, H - fg.height), mk)
text(shade(bg, 0.32, 170), H * 0.075, W, H).save(f"{out}/封面_9比16_1080x1920.jpg", quality=95)
print("done", out)
