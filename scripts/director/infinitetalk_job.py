"""用法：infinitetalk_job.py 素材目录 [段数]。ComfyUI 原生 InfiniteTalk 两人对口型（按官方模板 video_wan2_1_infinitetalk 的连线）。
注意：原生节点两人模式写死为 "add"——第一个人的音频整段放完才接第二个人，所以只能“A 说完、B 再说”，不能轮流。
每段 81 帧（25fps），后面每段用前面累计的画面做 previous_frames（偏移=累计帧数-9），截掉开头 9 帧动作参考后拼接。"""
import json, os, sys, subprocess, urllib.request
H = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/"); D = sys.argv[1]; sub = "wfl/it"  # D 里放 start.png、mask_woman.png、mask_man.png、woman.wav、man.wav、prompt.txt
def up(f):
    subprocess.run(["curl", "-s", "-F", f"image=@{os.path.join(D, f)};filename={f}", "-F", "type=input", "-F", f"subfolder={sub}", "-F", "overwrite=true", H + "/upload/image"], check=True, capture_output=True)
    return f"{sub}/{f}"
for f in ["start.png", "mask_woman.png", "mask_man.png", "woman.wav", "man.wav"]: up(f)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3; seed = 7; W, Hh = 480, 832
prompt = open(os.path.join(D, "prompt.txt")).read().strip()
p = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "ModelPatchLoader", "inputs": {"name": "wan2.1_infiniteTalk_multi_fp16.safetensors"}},
 "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
 "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
 "7": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "8": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "wav2vec2-chinese-base_fp16.safetensors"}},
 "9": {"class_type": "LoadAudio", "inputs": {"audio": f"{sub}/woman.wav"}},
 "10": {"class_type": "LoadAudio", "inputs": {"audio": f"{sub}/man.wav"}},
 "11": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["8", 0], "audio": ["9", 0]}},
 "12": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["8", 0], "audio": ["10", 0]}},
 "13": {"class_type": "LoadImage", "inputs": {"image": f"{sub}/start.png"}},
 "14": {"class_type": "LoadImage", "inputs": {"image": f"{sub}/mask_woman.png"}},
 "15": {"class_type": "LoadImage", "inputs": {"image": f"{sub}/mask_man.png"}},
 "16": {"class_type": "ImageToMask", "inputs": {"image": ["14", 0], "channel": "red"}},
 "17": {"class_type": "ImageToMask", "inputs": {"image": ["15", 0], "channel": "red"}},
 "18": {"class_type": "AudioConcat", "inputs": {"audio1": ["9", 0], "audio2": ["10", 0], "direction": "after"}},
}
acc = None
for k in range(N):
    t = f"c{k}"
    inp = {"mode": "two_speakers", "mode.audio_encoder_output_2": ["12", 0], "mode.mask_1": ["16", 0], "mode.mask_2": ["17", 0],
           "model": ["2", 0], "model_patch": ["3", 0], "positive": ["5", 0], "negative": ["6", 0], "vae": ["7", 0],
           "width": W, "height": Hh, "length": 81, "audio_encoder_output_1": ["11", 0], "motion_frame_count": 9, "audio_scale": 1.0, "start_image": ["13", 0]}
    if acc: inp["previous_frames"] = acc
    p[t] = {"class_type": "WanInfiniteTalkToVideo", "inputs": inp}
    p[t + "g"] = {"class_type": "CFGGuider", "inputs": {"model": [t, 0], "positive": [t, 1], "negative": [t, 2], "cfg": 1.0}}
    p[t + "k"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    p[t + "s"] = {"class_type": "BasicScheduler", "inputs": {"model": [t, 0], "scheduler": "normal", "steps": 6, "denoise": 1.0}}
    p[t + "n"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + k}}
    p[t + "x"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [t + "n", 0], "guider": [t + "g", 0], "sampler": [t + "k", 0], "sigmas": [t + "s", 0], "latent_image": [t, 3]}}
    p[t + "d"] = {"class_type": "VAEDecode", "inputs": {"samples": [t + "x", 0], "vae": ["7", 0]}}
    if acc is None:
        acc = [t + "d", 0]
    else:
        p[t + "t"] = {"class_type": "ImageFromBatch", "inputs": {"image": [t + "d", 0], "batch_index": 9, "length": 72}}
        p[t + "b"] = {"class_type": "ImageBatch", "inputs": {"image1": acc, "image2": [t + "t", 0]}}
        acc = [t + "b", 0]
p["90"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": acc, "audio": ["18", 0], "frame_rate": 25, "loop_count": 0, "filename_prefix": "wfl/infinitetalk_duo",
           "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False, "trim_to_audio": True, "pingpong": False, "save_output": True}}
json.dump(p, open(os.path.join(D, "infinitetalk_api.json"), "w"), ensure_ascii=False, indent=1)
if "--check" in sys.argv: print("built", len(p)); sys.exit()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(H + "/prompt", json.dumps({"prompt": p, "client_id": "mac-claude"}).encode(), {"Content-Type": "application/json"})))
    print("prompt_id", r.get("prompt_id"))
except urllib.error.HTTPError as e:
    print("ERROR", json.dumps(json.load(e), ensure_ascii=False)[:2500])
