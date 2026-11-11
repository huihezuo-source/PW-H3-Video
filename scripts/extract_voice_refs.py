"""从原片取各角色的原声参考音：demucs 分离人声 → 按时间段切句 → 每个角色拼成一条。

用法：python3 extract_voice_refs.py 原片.mp4 segments.json out_dir
segments.json：{"hero": [[32.85, 34.3], [61.5, 62.5]], "villain": [[80.3, 89.4]]}
  时间段先对人声轨跑 asr 的词级时间戳找；每段留一点余量，脚本会按响度修掉首尾。
输出：out_dir/vocals.wav（整条人声）和 out_dir/ref_<角色>.wav（句子之间留 0.35 秒静音）。
依赖：pip install demucs soundfile（第一次运行会下载 htdemucs 模型）。
出来后用 asr.py 转写每条 ref：字对不上（比如背景歌声混进来）的角色不要拿去克隆。
"""
import json, os, subprocess, sys
import numpy as np, soundfile as sf, torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

src, segs, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
os.makedirs(out, exist_ok=True)
wav_path = os.path.join(out, "orig.wav")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", wav_path], check=True)

m = get_model("htdemucs"); m.eval()
wav, sr = sf.read(wav_path, dtype="float32")
x = torch.tensor(wav.T)[None]
ref = x.mean(1)
x = (x - ref.mean()) / ref.std()
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    stems = apply_model(m, x, device=dev, shifts=1, split=True, overlap=0.25, progress=False)[0]
voc = (stems[m.sources.index("vocals")] * ref.std() + ref.mean()).T.numpy()
sf.write(os.path.join(out, "vocals.wav"), voc, sr)

y = voc.mean(1)
gap = np.zeros(int(0.35 * sr))
for name, ws in segs.items():
    parts = []
    for a, b in ws:
        c = y[int(a * sr):int(b * sr)]
        hop = int(0.02 * sr)
        e = np.array([np.sqrt(np.mean(c[i:i + hop] ** 2)) for i in range(0, len(c) - hop, hop)])
        on = np.where(e > e.max() * 0.08)[0]
        s, t = max(0, on[0] * hop - int(0.06 * sr)), min(len(c), (on[-1] + 1) * hop + int(0.08 * sr))
        c = c[s:t].copy()
        f = int(0.03 * sr)
        c[:f] *= np.linspace(0, 1, f); c[-f:] *= np.linspace(1, 0, f)
        parts += [c, gap]
    o = np.concatenate(parts[:-1])
    sf.write(os.path.join(out, f"ref_{name}.wav"), o / np.abs(o).max() * 0.9, sr)
    print(f"ref_{name}.wav {len(o) / sr:.1f}s")
