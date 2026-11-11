"""VoxCPM2 按文字描述设计角色声音，每个角色出几条试听，挑中的一条以后当参考音用。

用法：python3 voice_design.py voices.json [--wait out_dir]
voices.json：
{
  "prefix": "voice_design",        # ComfyUI 输出子目录
  "seed": 301, "takes": 2,         # 每个角色出几条，种子依次加 1
  "voices": {
    "sanhu": {"design": "四十岁左右的中年男警察，嗓音低沉沙哑……", "text": "你们是什么人？……"}
  }
}
描述写法：身份 + 音色 + 情绪 + 语速，一句话说清，不要堆标签。
几个角色之间要拉开年龄、音高和语速，观众才分得清谁在说话。
试听文本最好 6–10 秒，用角色自己的台词；挑中的那条以后给 H3 当参考音、给后期配音克隆。
"""
import json, sys
import comfy

cfg = json.load(open(sys.argv[1]))
p, i = {}, 0
for name, v in cfg["voices"].items():
    for t in range(cfg.get("takes", 2)):
        seed = cfg.get("seed", 301) + t
        a, b = str(10 + i * 2), str(11 + i * 2)
        p[a] = {"class_type": "VoxCPM_TTS", "inputs": {"model_name": "VoxCPM2", "lora_name": "None", "text": v["text"],
                "voice_design": v["design"], "cfg": v.get("cfg", 2.0), "steps": 10, "normalize_text": False,
                "seed": seed, "force_offload": True, "device": "cuda", "dtype": "auto"}}
        p[b] = {"class_type": "SaveAudio", "inputs": {"audio": [a, 0],
                "filename_prefix": f"{cfg.get('prefix', 'voice_design')}/{name}_s{seed}"}}
        i += 1
if "--wait" in sys.argv:
    print(comfy.run(p, sys.argv[sys.argv.index("--wait") + 1]))
else:
    print("prompt_id", comfy.submit(p))
