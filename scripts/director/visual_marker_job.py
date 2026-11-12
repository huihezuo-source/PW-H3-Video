"""T8 Visual Marker 最小接法：Prepare 在参考图上画框 → Prompt 生成解释框的提示词 → 普通 H3 Ref2VA（DMAD 8 步 + lcm），用来打样“谁从哪走到哪”。
用法：visual_marker_job.py job.json [seed]；job：{"ref": 参考图, "markers": {...T8 格式...}, "base_prompt": 场景描述, "size": [576, 800], "length": 121, "name": "vm"}
T8 规定每个人物框都要用 relation 连到一个目标框；不动的角色不画框，写进场景描述。目标框要画在认得出的物体上（门、桌子），不要画空地：H3 会重新构图。"""
import json, os, sys, subprocess, urllib.request
H = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
job = json.load(open(sys.argv[1]))
ref = job["ref"]
subprocess.run(["curl", "-s", "-F", f"image=@{ref};filename=vm_dogs_ref.png", "-F", "type=input", "-F", "subfolder=wfl/vm", "-F", "overwrite=true", H + "/upload/image"], check=True, capture_output=True)
markers = job["markers"]
base = job["base_prompt"]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
p = {
 "10": {"class_type": "LoadImage", "inputs": {"image": "wfl/vm/vm_dogs_ref.png"}},
 "60": {"class_type": "MiniMaxH3VisualMarkerPrepareEXPT8", "inputs": {"image": ["10", 0], "mode": "render_rectangles", "markers_json": json.dumps(markers), "line_width": 4, "expected_source_sha256": ""}},
 "61": {"class_type": "MiniMaxH3VisualMarkerPromptEXPT8", "inputs": {"marker_plan": ["60", 1], "base_prompt": base, "mode": "generated", "picture_ordinal": 1, "include_reminder": True}},
 "62": {"class_type": "PreviewAny", "inputs": {"source": ["61", 0]}},
 "63": {"class_type": "SaveImage", "inputs": {"images": ["60", 0], "filename_prefix": "wfl/vm_marked"}},
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "1b": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "dmad_minimax_h3_8step_full_critic_comfyui.safetensors", "strength_model": 1.0}},
 "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1b", 0], "shift_video": 12, "shift_audio": 2}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "30": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["3", 0], "prompt": ["61", 0], "width": job.get("size", [576, 800])[0], "height": job.get("size", [576, 800])[1], "length": job.get("length", 121), "ref_image_size": "match",
        "vae": ["4", 0], "audio_vae": ["5", 0], "ref_images.ref_image_0": ["60", 0]}},
 "40": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["30", 0]}},
 "41": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
 "42": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
 "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
 "44": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["43", 0], "guider": ["40", 0], "sampler": ["41", 0], "sigmas": ["42", 0], "latent_image": ["30", 1]}},
 "45": {"class_type": "VAEDecode", "inputs": {"samples": ["44", 0], "vae": ["4", 0]}},
 "46": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["44", 0], "vae": ["5", 0]}},
 "47": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["45", 0], "audio": ["46", 0], "frame_rate": 24, "loop_count": 0, "filename_prefix": f"wfl/{job.get('name', 'vm')}_s{seed}",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}},
}
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(H + "/prompt", json.dumps({"prompt": p, "client_id": "mac-claude"}).encode(), {"Content-Type": "application/json"})))
    print("prompt_id", r.get("prompt_id"))
except urllib.error.HTTPError as e:
    print("ERROR", json.dumps(json.load(e), ensure_ascii=False)[:2000])
