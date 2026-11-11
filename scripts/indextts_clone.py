"""调用家里电脑上 IndexTTS-2.5 的网页界面（Gradio），按每个角色的原声克隆新台词。

用法：python3 indextts_clone.py cast.json out_dir [--url http://电脑IP:7860]
cast.json（和 clone_cast.py 同格式，多了可选的情绪和语速）：
{
  "seed_takes": 2,                       # 每句出几条（IndexTTS 默认采样，每次结果略有不同）
  "cast": {
    "zhuli": {"ref": "原声ft/ref_zhuli.wav", "text": "快点抬轿。",
              "emo": [0,0.6,0,0,0.2,0,0,0],  # 可选，8 维情绪：喜 怒 哀 惧 厌恶 低落 惊讶 平静；不写就跟随参考音的情绪
              "speed": 1.0}                   # 可选，>1 变慢
  }
}
电脑端启动：cd D:\\index-tts; uv run webui.py   （默认监听 0.0.0.0:7860）
依赖：pip install gradio_client
"""
import json, os, shutil, sys
from gradio_client import Client, handle_file

cfg, out = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2]
url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://100.122.217.58:7860"
os.makedirs(out, exist_ok=True)
c = Client(url, verbose=False)
base = os.path.dirname(os.path.abspath(sys.argv[1]))
for name, v in cfg["cast"].items():
    ref = v["ref"] if os.path.isabs(v["ref"]) else os.path.join(base, v["ref"])
    emo = v.get("emo")
    for k in range(cfg.get("seed_takes", 2)):
        res = c.predict(
            2 if emo else 0,                             # 情绪控制：0 跟随参考音，2 用情绪向量
            handle_file(ref), v["text"], "ZH",
            None, v.get("emo_weight", 0.8),
            *(emo or [0.0] * 8),
            "", False,                                   # 情绪文字、随机采样（随机会降低相似度，关掉）
            120, float(v.get("speed", 1.0)),
            True, 0.8, 30, 0.8, 0.0, 3, 10.0, 1500,      # do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens
            api_name="/gen_single")
        path = res["value"] if isinstance(res, dict) else res
        dst = os.path.join(out, f"{name}_{k + 1}.wav")
        shutil.copy(path, dst)
        print(dst, flush=True)
