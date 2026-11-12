"""用 AICG3D 短剧工作台的“无限分段”节点出片：H3Loader → SigmaShift → 注意力 → SequenceGlobal →
SequenceSegment×N（第一段接素材和声音库，后面的接上一段）→ SequenceCombine。H3 用声音库里的参考音色自己念台词（生成音频模式）。
job.json：{"name", "resolution": "540P", "aspect": "9:16", "seed", "images": [本地路径...], "voices": [{"seat": "girl", "name", "ref": 本地音频, "picture_index": N, "age", "gender"}],
          "segments": [{"seconds", "prompt"}]}  座位顺序 girl/boy/heroine/hero/old_woman/old_man 对应 (S1)…(S6)。"""
import json, os, sys, subprocess, urllib.request
H = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")  # 跑 Songssx / AICG3D / SAM3 节点的 ComfyUI
job = json.load(open(sys.argv[1])); sub = f"wfl/{job['name']}"
def up(path, name):
    subprocess.run(["curl", "-s", "-F", f"image=@{path};filename={name}", "-F", "type=input", "-F", f"subfolder={sub}", "-F", "overwrite=true", H + "/upload/image"], check=True, capture_output=True)
    return f"{sub}/{name}"
imgs = [{"filename": up(p, f"pic{i + 1}{os.path.splitext(p)[1]}")} for i, p in enumerate(job["images"])]
roles = []
for v in job["voices"]:
    ref = up(v["ref"], f"voice_{v['seat']}{os.path.splitext(v['ref'])[1]}")
    roles.append({"id": v["seat"], "name": v["name"], "mode": "reference", "description": v.get("description", v["name"]), "language": "Chinese",
                  "age": v.get("age", "adult"), "gender": v.get("gender", "male"), "pitch": "natural", "pace": "natural",
                  "picture_index": v["picture_index"], "reference": ref, "reference_start": 0, "reference_duration": 0, "rights_confirmed": True})
lora = job.get("lora", "dmad_minimax_h3_8step_full_critic_comfyui.safetensors")
p = {
 "1": {"class_type": "AICG3D_H3Loader", "inputs": dict({"fl2va_model": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "ref2va_model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "video_vae": "minimax_h3_video_vae_fp16.safetensors", "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "lora_1": lora, "lora_1_strength": 1.0, "non_h3_policy": "自动回落", "注意力加速(Sage)": "Comfy Kitchen INT8"},
        **{f"lora_{k}": "none" for k in range(2, 10)}, **{f"lora_{k}_strength": 1.0 for k in range(2, 10)})},
 "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 1], "shift_video": 12, "shift_audio": 2}},
 "3": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["2", 0], "attention": "comfy kitchen attention"}},
 "5": {"class_type": "AICG3D_H3SequenceGlobal", "inputs": {"h3_bundle": ["1", 0], "model": ["3", 0], "resolution": job.get("resolution", "540P"), "aspect_ratio": job.get("aspect", "9:16"),
        "fps": 24.0, "handoff_frames": 22, "handoff_mode": "尾帧续写（latent）", "ref_image_size": "match", "sampler_name": "lcm", "scheduler": "simple", "steps": job.get("steps", 8),
        "denoise": 1.0, "vram_policy": "释放缓存", "sample_preview": "关", "preview_interval": 1, "prompt_optimizer_settings": False, "audio_mode": "生成音频",
        "vram_tier": "自动（按显卡）", "exposure_lock": "跨段曝光锁定", "keep_full": True, "second_pass": False,
        "second_pass_model": "(place models in: D:\\ComfyUI-oj\\models\\latent_upscale_models)", "second_pass_scale": 1.5, "second_pass_sampler_name": "euler",
        "second_pass_scheduler": "simple", "second_pass_steps": 4, "second_pass_denoise": 0.35}},
 "6": {"class_type": "AICG3D_H3MediaLoader", "inputs": {"media_state": json.dumps({"images": imgs, "audios": [], "videos": []}, ensure_ascii=False), "视频解码长边": 0}},
 "7": {"class_type": "AICG3D_H3CharacterVoiceLibrary", "inputs": {"voice_state": json.dumps({"version": 1, "roles": roles}, ensure_ascii=False), "enabled": True}},
}
prev = None
for i, s in enumerate(job["segments"]):
    nid = f"s{i + 1}"
    inp = {"prompt": s["prompt"], "seconds": s["seconds"], "seed": job.get("seed", 7) + i, "guide_frame_idx": 0}
    if prev is None: inp.update({"sequence_config": ["5", 0], "media": ["6", 0], "voice_cast": ["7", 0]})
    else: inp["previous_segment"] = [prev, 0]
    p[nid] = {"class_type": "AICG3D_H3SequenceSegment", "inputs": inp}; prev = nid
p["9"] = {"class_type": "AICG3D_H3SequenceCombine", "inputs": {"final_segment": [prev, 0], "filename_prefix": f"wfl/{job['name']}"}}
json.dump(p, open(sys.argv[1].replace(".json", "_api.json"), "w"), ensure_ascii=False, indent=1)
if "--check" in sys.argv: print("built", len(p)); sys.exit()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(H + "/prompt", json.dumps({"prompt": p, "client_id": "mac-claude"}).encode(), {"Content-Type": "application/json"})))
    print("prompt_id", r.get("prompt_id"))
except urllib.error.HTTPError as e:
    d = json.load(e); print("ERROR", json.dumps(d, ensure_ascii=False)[:2500])
