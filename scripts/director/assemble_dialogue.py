"""把每句配音拼成一条对白音轨，输出 dialogue.wav 和 lines.json（每句谁说、起止秒数）。
spec.json：{"sr": 48000, "lead": 0.3, "tail": 0.3, "lines": [{"who": "徒弟", "file": "x.flac", "part": 0, "parts": 1, "gap": 0.4, "text": "..."}]}
parts>1 时按最长的静音把一条录音切成几段，取第 part 段。每句前后自动去掉静音；gap 是这句之后停多久；"mix_with_prev": true 表示和上一句同时播放（比如一起笑）。"""
import json, sys, subprocess, numpy as np
spec = json.load(open(sys.argv[1])); out = sys.argv[2]; SR = spec.get("sr", 48000)
def load(f):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)
def active(a, thr=0.02, win=0.03):
    n = int(SR * win); e = np.sqrt(np.convolve(a ** 2, np.ones(n) / n, "same")); return e > thr
def trim(a):
    m = active(a); idx = np.nonzero(m)[0]
    if not len(idx): return a
    return a[max(0, idx[0] - int(0.05 * SR)):idx[-1] + int(0.08 * SR)]
def split(a, parts):
    m = active(a); gaps = []; i = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]: j += 1
            if i > 0 and j < len(m): gaps.append((j - i, (i + j) // 2))
            i = j
        else: i += 1
    cuts = sorted(c for _, c in sorted(gaps, reverse=True)[:parts - 1])
    return np.split(a, cuts)
t = spec.get("lead", 0.3); track = np.zeros(int(SR * 60), np.float32); lines = []; prev_start = 0
for L in spec["lines"]:
    a = load(L["file"])
    if L.get("parts", 1) > 1: a = split(a, L["parts"])[L["part"]]
    a = trim(a)
    start = prev_start if L.get("mix_with_prev") else t
    s = int(start * SR); track[s:s + len(a)] += a * L.get("gain", 1.0)
    end = start + len(a) / SR
    lines.append({"who": L["who"], "text": L.get("text", ""), "start": round(start, 2), "end": round(end, 2)})
    prev_start = start; t = max(t, end) + L.get("gap", 0.4)
total = t - lines[-1:][0].get("gap", 0) + spec.get("tail", 0.3) if False else max(l["end"] for l in lines) + spec.get("tail", 0.3)
track = track[:int(total * SR)]; peak = np.abs(track).max(); track = track / peak * 0.89 if peak > 0 else track
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-ac", "2", out], input=track.tobytes(), check=True)
json.dump({"duration": round(total, 2), "lines": lines}, open(out.rsplit(".", 1)[0] + "_lines.json", "w"), ensure_ascii=False, indent=1)
print(out, round(total, 2), "s"); [print(" ", l) for l in lines]
