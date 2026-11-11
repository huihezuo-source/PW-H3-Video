"""Qwen-Image 批量出角色母版：每个角色 = 风格前缀 + 角色描述，每人出 batch 张挑一张。

用法：python3 t2i_masters.py masters.json [--wait out_dir]
masters.json：
{
  "style": "真实的电影剧照，……（基调与风格前缀）",
  "negative": "卡通,动画,插画,……",
  "prefix": "masters",           # ComfyUI 输出子目录
  "seed": 9300, "batch": 2, "width": 896, "height": 1232,
  "characters": {"tang": "一位……", "wukong": "一只……"}
}
"""
import json, sys
import comfy

cfg = json.load(open(sys.argv[1]))
p = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
    "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": cfg.get("negative", "")}},
    "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": cfg.get("width", 896), "height": cfg.get("height", 1232),
                                                          "batch_size": cfg.get("batch", 2)}},
}
i = 10
for name, desc in cfg["characters"].items():
    k = lambda j: str(i + j)
    p[k(0)] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": cfg["style"] + desc}}
    p[k(1)] = {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": [k(0), 0], "negative": ["6", 0],
               "latent_image": ["7", 0], "seed": cfg.get("seed", 9300) + i, "steps": 30, "cfg": 3.0,
               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    p[k(2)] = {"class_type": "VAEDecode", "inputs": {"samples": [k(1), 0], "vae": ["3", 0]}}
    p[k(3)] = {"class_type": "SaveImage", "inputs": {"images": [k(2), 0], "filename_prefix": f"{cfg.get('prefix', 'masters')}/{name}"}}
    i += 10
if "--wait" in sys.argv:
    print(comfy.run(p, sys.argv[sys.argv.index("--wait") + 1]))
else:
    print("prompt_id", comfy.submit(p))
