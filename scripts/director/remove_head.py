"""把角色照片抠成“无头照”：SAM 3.1按文字找头部（含帽子、头饰、头发），
遮罩稍微外扩，头部区域填成背景的中性灰，脖子以下保留。用法：remove_head.py in.png out.png "dog head" [--grow 0.04]
资产规矩：头像单独做、身体照去掉头，出视频时脸只从头像取，远景近景的脸才一致。"""
import json, os, sys, time, urllib.request, subprocess
import numpy as np
from PIL import Image, ImageFilter
H = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")  # 跑 Songssx / AICG3D / SAM3 节点的 ComfyUI
src, dst, text = sys.argv[1], sys.argv[2], sys.argv[3]
grow = float(sys.argv[sys.argv.index("--grow") + 1]) if "--grow" in sys.argv else 0.04
name = "rh_" + os.path.basename(src).encode().hex()[:24] + ".png"
subprocess.run(["curl", "-s", "-F", f"image=@{src};filename={name}", "-F", "type=input", "-F", "overwrite=true", H + "/upload/image"], check=True, capture_output=True)
p = {
 "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
 "2": {"class_type": "LoadImage", "inputs": {"image": name}},
 "5": {"class_type": "MaskToImage", "inputs": {"mask": ["m0", 0]}},
 "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "headmask/m"}},
}
# 多个文字用 | 分开（例如 "dog head|crown"），各自检测后合并
for k, t in enumerate(text.split("|")):
    p[f"t{k}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": t.strip(), "clip": ["1", 1]}}
    p[f"d{k}"] = {"class_type": "SAM3_Detect", "inputs": {"model": ["1", 0], "image": ["2", 0], "threshold": 0.35, "refine_iterations": 2, "individual_masks": False, "conditioning": [f"t{k}", 0]}}
    p[f"m{k}"] = {"class_type": "MaskComposite", "inputs": {"destination": [f"m{k-1}", 0], "source": [f"d{k}", 0], "x": 0, "y": 0, "operation": "add"}} if k else \
                 {"class_type": "ThresholdMask", "inputs": {"mask": [f"d{k}", 0], "value": 0.5}}
last = len(text.split("|")) - 1
p["5"]["inputs"]["mask"] = [f"m{last}", 0]
req = urllib.request.Request(H + "/prompt", json.dumps({"prompt": p, "client_id": "mac-claude"}).encode(), {"Content-Type": "application/json"})
pid = json.load(urllib.request.urlopen(req))["prompt_id"]
for _ in range(240):
    h = json.load(urllib.request.urlopen(f"{H}/history/{pid}"))
    if pid in h: break
    time.sleep(2)
st = h[pid]["status"]
if st.get("status_str") == "error": raise SystemExit(json.dumps(st["messages"])[-800:])
f = h[pid]["outputs"]["6"]["images"][0]
mpath = os.path.join(__import__("tempfile").gettempdir(), "headmask_" + name)
urllib.request.urlretrieve(f"{H}/view?filename={f['filename']}&subfolder={f['subfolder']}&type=output", mpath)
img = Image.open(src).convert("RGB"); m = Image.open(mpath).convert("L").resize(img.size)
a = np.array(m) > 127
if a.sum() == 0: raise SystemExit("no head found: " + text)
ys, xs = np.nonzero(a); hh = ys.max() - ys.min()
r = max(3, int(hh * grow))
m = m.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(r * 2 + 1)).filter(ImageFilter.GaussianBlur(r * 0.4))
arr = np.array(img).astype(np.float32); mm = np.array(m).astype(np.float32)[..., None] / 255
# 用背景（图像上边缘一圈）的颜色填，而不是纯灰，避免一块突兀的色块
# 用头顶上方那一块背景的颜色填（左右边缘常有暗角，取它会留下深色剪影）；按行混合头部两侧背景，保留上下的明暗渐变
x0, x1 = xs.min(), xs.max(); top = ys.min()
band = arr[max(0, top - 60):max(1, top - 8), x0:x1 + 1] if top > 20 else arr[:20, x0:x1 + 1]
fill = np.median(band.reshape(-1, 3), 0)
side = np.concatenate([arr[:, max(0, x0 - 80):max(1, x0 - 20)], arr[:, x1 + 20:x1 + 80]], axis=1)
row_col = np.median(side, axis=1) if side.shape[1] else np.tile(fill, (arr.shape[0], 1))
blend = np.clip((np.arange(arr.shape[0]) - top) / max(1, hh), 0, 1)[:, None]
fill_rows = (fill * (1 - blend) + row_col * blend)[:, None, :]
out = arr * (1 - mm) + fill_rows * mm
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(dst)
print(dst, "head box h", int(hh), "fill", fill.round().tolist())
