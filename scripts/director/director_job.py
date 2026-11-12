"""用 Songssx 的 MiniMax H3 时间线导演台（Material Planner + Finite Segment Sampler）出片。
job.json：{"name", "size": [w, h], "seconds", "seed", "images": [{"id", "file"(本地路径)}], "audio": 本地 wav（锁定原声）,
          "segments": [{"start", "end", "images": [id...], "prompt"}]}
每段帧数 5+17n，相邻段要有对齐的重叠（作者示例 39 帧）。素材上传到 ComfyUI input/wfl/<name>/。
采样：H3 Ref2VA int8 + DMAD 8 步 LoRA + lcm 8 步（DMAD 作者的工作点）。用法：director_job.py job.json [--check]"""
import json, os, sys, subprocess, urllib.request
from PIL import Image
H = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")  # 跑 Songssx / AICG3D / SAM3 节点的 ComfyUI
job = json.load(open(sys.argv[1])); sub = f"wfl/{job['name']}"
def up(path, name):
    subprocess.run(["curl", "-s", "-F", f"image=@{path};filename={name}", "-F", "type=input", "-F", f"subfolder={sub}", "-F", "overwrite=true", H + "/upload/image"], check=True, capture_output=True)
    return f"{sub}/{name}"
images = []
for im in job["images"]:
    w, h = Image.open(im["file"]).size
    images.append({"id": im["id"], "file": up(im["file"], im["id"] + ".png"), "name": im["id"] + ".png", "width": w, "height": h})
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", job["audio"]], capture_output=True, text=True).stdout)
audio = {"id": "dialogue", "file": up(job["audio"], "dialogue.wav"), "name": "dialogue.wav", "duration": dur, "trimStart": 0, "audioMode": "locked"}
segs = [{"images": s["images"], "audios": ["dialogue"], "prompt": s["prompt"], "startFrame": s["start"], "endFrame": s["end"]} for s in job["segments"]]
timeline = {"version": 9, "fps": 24, "globalPrompt": "", "secondPass": False, "selection": {"start": 0, "duration": job["seconds"]},
            "videoAudioEnabled": True, "videoClips": [], "images": images, "audios": [audio],
            "segmentConfig": {"count": len(segs), "activeIndex": 0, "mode": "timeline", "segments": segs}}
W, Hh = job["size"]
p = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "1b": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": job.get("lora", "dmad_minimax_h3_8step_full_critic_comfyui.safetensors"), "strength_model": 1.0}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "8": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "11": {"class_type": "MiniMaxH3TimelinePlanner", "inputs": {"width": W, "height": Hh, "generation_seconds": job["seconds"], "timeline_data": json.dumps(timeline, ensure_ascii=False)}},
 "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": job.get("sampler", "lcm")}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": ["1b", 0], "scheduler": "simple", "steps": job.get("steps", 8), "denoise": 1.0}},
 "30": {"class_type": "MiniMaxH3FiniteSegmentSampler", "inputs": {"model": ["1b", 0], "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["8", 0], "finite_plan": ["11", 2],
        "sampler": ["13", 0], "sigmas": ["9", 0], "seed": job.get("seed", 7), "continue_audio_latent": True, "ref_image_size": "match"}},
 "29": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["30", 1], "audio": ["30", 2], "frame_rate": 24, "loop_count": 0, "filename_prefix": f"wfl/{job['name']}",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}},
 "31": {"class_type": "PreviewAny", "inputs": {"source": ["30", 3]}},
}
json.dump(p, open(sys.argv[1].replace(".json", "_api.json"), "w"), ensure_ascii=False, indent=1)
if "--check" in sys.argv: print("built", len(p)); sys.exit()
r = json.load(urllib.request.urlopen(urllib.request.Request(H + "/prompt", json.dumps({"prompt": p, "client_id": "mac-claude"}).encode(), {"Content-Type": "application/json"})))
print("prompt_id", r.get("prompt_id"), r.get("node_errors") or "")
