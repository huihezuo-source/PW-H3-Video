"""MiniMax-H3-Image 改图 / 多图参考出图（ComfyUI 0.45+，官方模板去掉扩写后的最小版）。

用法：python3 h3_image.py "英文提示词" --img 图1.png [--img 图2.png ...] [--size 1728x2368] [--seeds 1,2] [--prefix h3img/x] [--wait out_dir]

- 图1 是 <Picture 1>（要改的那张，也决定构图参考），后面依次是 <Picture 2>…，最多 16 张。
- 尺寸：400–2000 万像素有效，低于 400 万明显变差。3:4 竖图 400 万 = 1728x2368，1600 万 = 3392x4672。
- 提示词结构（官方改图写法）：The X in <Picture 1> is <Subject 1>. The target image is …. <Subject 1> is partially preserved: … remain, while …. … from <Picture 1> are removed. 最后写光的方向和影子。
- 不要拿 H3 Image 出的图再改：一两轮后脸上起纹理、整图发花。要改几处就从原图一次写全。
- 6 步、cfg 1、euler/simple，是官方模板的设置。
"""
import os, sys
import comfy

arg = lambda f, d=None: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d
prompt = sys.argv[1]
imgs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--img"]
w, h = map(int, arg("--size", "1728x2368").split("x"))
seeds = [int(x) for x in arg("--seeds", "1").split(",")]
prefix = arg("--prefix", "h3img/out")

for seed in seeds:
    p = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": os.environ.get("H3_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"), "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"}},
    }
    cond = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": w, "height": h}
    for i, im in enumerate(imgs):
        name = comfy.upload(im, f"h3img_{i}_{os.path.basename(im)}")
        p[str(10 + i * 2)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        # 参考图先放大到和目标差不多的像素，低分辨率输入效果差（官方模板放到 400 万）
        p[str(11 + i * 2)] = {"class_type": "ImageScale", "inputs": {"image": [str(10 + i * 2), 0], "upscale_method": "lanczos",
                                                                    "width": w, "height": h, "crop": "center"}} if i == 0 else \
                             {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": [str(10 + i * 2), 0], "upscale_method": "lanczos",
                                                                                 "megapixels": w * h / 1e6, "resolution_steps": 32}}
        cond[f"images.image_{i + 1}"] = [str(11 + i * 2), 0]
    p["40"] = {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond}
    p["41"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["40", 0]}}
    p["42"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["40", 0], "negative": ["41", 0], "latent_image": ["40", 1],
                                                    "seed": seed, "steps": int(arg("--steps", 6)), "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    p["43"] = {"class_type": "VAEDecode", "inputs": {"samples": ["42", 0], "vae": ["3", 0]}}
    p["44"] = {"class_type": "SaveImage", "inputs": {"images": ["43", 0], "filename_prefix": f"{prefix}_s{seed}"}}
    if "--wait" in sys.argv:
        print(comfy.run(p, arg("--wait")))
    else:
        print("prompt_id", comfy.submit(p))
