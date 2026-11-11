"""VoxCPM2 按每个角色自己的参考音克隆，一次出全部角色的新台词。

用法：python3 clone_cast.py cast.json [--wait out_dir]
cast.json：
{
  "prefix": "cast", "seed": 401,
  "modes": ["isolated", "continuation"],   # isolated 只给参考音；continuation 再加参考音原文，学得更像但原文必须一字不差
  "cast": {
    "hero": {"ref": "ref_hero.wav", "ref_text": "参考音里说的原文", "text": "要说的新台词"}
  }
}
参考音从原片取时：先用 demucs 分离人声，按台词时间切出这个角色的几句，中间留 0.35 秒静音拼起来，5–10 秒最好。
背景里有带人声的歌、或只剩一两个字的角色，取不出干净参考音，改用 voice_design.py 设计。
"""
import json, os, sys
import comfy

cfg = json.load(open(sys.argv[1]))
modes = cfg.get("modes", ["isolated", "continuation"])
p, i = {}, 0
for name, c in cfg["cast"].items():
    ref = comfy.upload(c["ref"], f"cast_{name}{os.path.splitext(c['ref'])[1]}")
    load = str(100 + i)
    p[load] = {"class_type": "LoadAudio", "inputs": {"audio": ref}}
    for k, mode in enumerate(modes):
        vc, tts, save = str(200 + i * 10 + k * 3), str(201 + i * 10 + k * 3), str(202 + i * 10 + k * 3)
        inp = {"trim_silence": True, "reference_audio": [load, 0]}
        if mode == "continuation":
            inp.update({"prompt_audio": [load, 0], "prompt_text": c["ref_text"]})
        p[vc] = {"class_type": "VoxCPM_VoiceCloning", "inputs": inp}
        seed = cfg.get("seed", 401) + k
        p[tts] = {"class_type": "VoxCPM_TTS", "inputs": {"model_name": "VoxCPM2", "lora_name": "None", "text": c["text"],
                  "voice_design": "", "cfg": c.get("cfg", 2.0), "steps": 10, "normalize_text": False, "seed": seed,
                  "force_offload": True, "device": "cuda", "dtype": "auto", "voice_config": [vc, 0]}}
        p[save] = {"class_type": "SaveAudio", "inputs": {"audio": [tts, 0],
                   "filename_prefix": f"{cfg.get('prefix', 'cast')}/{name}_{mode}"}}
    i += 1
if "--wait" in sys.argv:
    print(comfy.run(p, sys.argv[sys.argv.index("--wait") + 1]))
else:
    print("prompt_id", comfy.submit(p))
