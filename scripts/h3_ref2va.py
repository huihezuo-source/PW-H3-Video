"""组装并提交 MiniMax H3 Ref2VA 工作流：参考图、参考音、首帧锚定。

用法：python3 h3_ref2va.py job.json [--wait out_dir]
job.json：
{
  "prompt": "按 h3-prompt-writing 官方格式写好的提示词",
  "width": 448, "height": 608, "length": 361,   # length = 秒数 × 24 + 1
  "steps": 20, "seed": 7, "prefix": "h3/seg1",
  "images": ["first_frame.png", "master_a.png"],  # 本地文件，自动上传
  "anchor_frames": [0],                          # 第 i 张图锚定到第几帧；首帧放第一张锚定到 0
  "guides": [],                                  # 可选，只锚定、不当参考图的帧：[{"image": "s2.png", "frame": 40}]，多镜头切点首帧用这个，省参考图 token
                                                 # 也可以锁一段台词音频：{"audio": "line1.wav", "frame": 14}，H3 只对口型，不自己生成这句的声音
  "audios": [],                                  # 参考音：说话角色的声音参考；旁白段留空
  "videos": [],                                  # 可选，参考视频（24fps、2–15 秒）：提供镜头运动、切镜和动作，提示词里叫 <Video 1>
  "shift": null,                                 # 可选 [video, audio]，一般不用
  "turbo": false,                                # true = 4 步加速 LoRA（minimax_h3_ref2v_turbo_4step_v0.1），euler/simple，shift 12/3，约快 3 倍
  "loras": [],                                   # 可选 [["h3_character_swap_v1.5.safetensors", 1.0]]：换人 LoRA，配 videos（原片）+ guides 锁音频
  "vsa": null                                    # 配合 turbo：稀疏注意力比例（如 0.75），要装 ComfyUI-Ref2VA-VSA 节点和 fasth3_vsa_gate.safetensors，且 PyTorch 必须是 CUDA 13 版（cu128 会报 Backend cuda disabled）
}
旁白段的 prompt 里不要写任何台词；一次生成只有一个发声来源。
"""
import json, os, sys
import comfy

a = json.load(open(sys.argv[1]))
p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}}}
model = ["1", 0]
for i, (ln, st) in enumerate(a.get("loras", [])):
    p[f"8{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": ln, "strength_model": st}}
    model = [f"8{i}", 0]
if a.get("turbo"):
    p["6"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}}
    model = ["6", 0]
    if a.get("vsa"):
        p["7"] = {"class_type": "Ref2VAVSAGatePatch", "inputs": {"model": model, "gate_file": "fasth3_vsa_gate.safetensors", "sparsity": a["vsa"]}}
        model = ["7", 0]
    a.setdefault("shift", [12, 3])
    a.setdefault("steps", 4)
if a.get("shift"):
    p["2"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": a["shift"][0], "shift_audio": a["shift"][1]}}
    model = ["2", 0]
p["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": os.environ.get("H3_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"), "type": "minimax", "device": "default"}}
p["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}}
p["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}}
r = {"clip": ["3", 0], "prompt": a["prompt"], "width": a["width"], "height": a["height"], "length": a["length"],
     "ref_image_size": "match", "vae": ["4", 0], "audio_vae": ["5", 0]}
for i, img in enumerate(a.get("images", [])):
    name = comfy.upload(img, f"h3_{i}_{os.path.basename(img)}")
    p[str(10 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
    r[f"ref_images.ref_image_{i}"] = [str(10 + i), 0]
for i, au in enumerate(a.get("audios", [])):
    name = comfy.upload(au, f"h3a_{i}_{os.path.basename(au)}")
    p[str(20 + i)] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
    r[f"ref_audios.ref_audio_{i}"] = [str(20 + i), 0]
for i, v in enumerate(a.get("videos", [])):
    name = comfy.upload(v, f"h3v_{i}_{os.path.basename(v)}")
    p[str(50 + i * 2)] = {"class_type": "LoadVideo", "inputs": {"file": name}}
    p[str(51 + i * 2)] = {"class_type": "GetVideoComponents", "inputs": {"video": [str(50 + i * 2), 0]}}
    r[f"ref_videos.ref_video_{i}"] = [str(51 + i * 2), 0]
p["30"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r}
pos, lat = ["30", 0], ["30", 1]
for i, fr in enumerate(a.get("anchor_frames", [])):
    nid = str(31 + i)
    p[nid] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": pos, "latent": lat, "frame_idx": fr, "vae": ["4", 0], "image": [str(10 + i), 0]}}
    pos = [nid, 0]
for i, g in enumerate(a.get("guides", [])):
    lid, nid = str(60 + i * 2), str(61 + i * 2)
    inp = {"positive": pos, "latent": lat, "frame_idx": g["frame"], "vae": ["4", 0]}
    if "image" in g:
        p[lid] = {"class_type": "LoadImage", "inputs": {"image": comfy.upload(g["image"], f"h3g_{i}_{os.path.basename(g['image'])}")}}
        inp["image"] = [lid, 0]
    else:
        p[lid] = {"class_type": "LoadAudio", "inputs": {"audio": comfy.upload(g["audio"], f"h3ga_{i}_{os.path.basename(g['audio'])}")}}
        inp.update({"audio": [lid, 0], "audio_vae": ["5", 0]})
    p[nid] = {"class_type": "MiniMaxH3AddGuide", "inputs": inp}
    pos = [nid, 0]
p["40"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": pos}}
p["41"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler" if a.get("turbo") else "res_multistep"}}
p["42"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": a.get("steps", 20), "denoise": 1.0}}
p["43"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": a.get("seed", 7)}}
p["44"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["43", 0], "guider": ["40", 0], "sampler": ["41", 0], "sigmas": ["42", 0], "latent_image": lat}}
p["45"] = {"class_type": "VAEDecode", "inputs": {"samples": ["44", 0], "vae": ["4", 0]}}
p["46"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["44", 0], "vae": ["5", 0]}}
p["47"] = {"class_type": "CreateVideo", "inputs": {"images": ["45", 0], "audio": ["46", 0], "fps": 24}}
p["48"] = {"class_type": "SaveVideo", "inputs": {"video": ["47", 0], "filename_prefix": a.get("prefix", "h3/out"), "format": "mp4", "codec": "h264"}}
if "--wait" in sys.argv:
    print(comfy.run(p, sys.argv[sys.argv.index("--wait") + 1], timeout=4 * 3600))
else:
    print("prompt_id", comfy.submit(p))
