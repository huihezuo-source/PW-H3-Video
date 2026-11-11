"""VoxCPM2 用一段参考音克隆音色，批量出旁白。所有旁白用同一段参考音，音色才统一。

用法：python3 tts_clone.py lines.json [--wait out_dir]
lines.json：
{
  "ref_audio": "voice_ref.flac",          # 本地参考音，6–10 秒最好
  "ref_text": "参考音里说的原文",
  "seed": 201,
  "lines": {"seg1": "第一句旁白……", "seg2": "第二句……"}
}
出来后用 asr.py 转写核对一字不差。
"""
import json, os, sys
import comfy

cfg = json.load(open(sys.argv[1]))
ref = comfy.upload(cfg["ref_audio"], "voice_ref" + os.path.splitext(cfg["ref_audio"])[1])
p = {
    "1": {"class_type": "LoadAudio", "inputs": {"audio": ref}},
    "2": {"class_type": "VoxCPM_VoiceCloning", "inputs": {"trim_silence": True, "prompt_text": cfg["ref_text"],
          "prompt_audio": ["1", 0], "reference_audio": ["1", 0]}},
}
for i, (name, text) in enumerate(cfg["lines"].items()):
    a, b = str(10 + i * 2), str(11 + i * 2)
    p[a] = {"class_type": "VoxCPM_TTS", "inputs": {"model_name": "VoxCPM2", "lora_name": "None", "text": text, "voice_design": "",
            "cfg": 2.0, "steps": 10, "normalize_text": False, "seed": cfg.get("seed", 201) + i, "force_offload": True,
            "device": "cuda", "dtype": "auto", "voice_config": ["2", 0]}}
    p[b] = {"class_type": "SaveAudio", "inputs": {"audio": [a, 0], "filename_prefix": f"voice/{name}"}}
if "--wait" in sys.argv:
    print(comfy.run(p, sys.argv[sys.argv.index("--wait") + 1]))
else:
    print("prompt_id", comfy.submit(p))
