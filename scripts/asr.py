"""faster-whisper 转写中文音频/视频，核对配音和台词表一字不差。

用法：python3 asr.py a.flac b.mp4 ...
依赖：pip install faster-whisper
"""
import sys
from faster_whisper import WhisperModel

m = WhisperModel("small", device="cpu", compute_type="int8")
for f in sys.argv[1:]:
    seg, info = m.transcribe(f, language="zh", condition_on_previous_text=False, vad_filter=False, word_timestamps=True)
    seg = list(seg)
    print(f.split("/")[-1], round(info.duration, 2), "".join(s.text for s in seg))
    print("  ", [(round(s.start, 1), round(s.end, 1)) for s in seg])
