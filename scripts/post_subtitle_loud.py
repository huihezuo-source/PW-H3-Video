"""后期：烧字幕（ASS）+ 两遍 loudnorm 到 -14 LUFS / -1.5 dBTP。

用法：python3 post_subtitle_loud.py in.mp4 out.mp4 subs.json [--fontsdir 字体目录]
subs.json：
{
  "font": "Hiragino Sans GB",
  "lines": [[开始秒, 结束秒, "字幕"], ...],        # 逐字来自台词表
  "pops":  [[开始秒, 结束秒, "大字"], ...]          # 可选：画面中间弹出的大字
}
字幕尺寸按视频分辨率自动缩放。
"""
import json, re, subprocess, sys

src, out, spec = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
fontsdir = sys.argv[sys.argv.index("--fontsdir") + 1] if "--fontsdir" in sys.argv else None
W, H = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                "-of", "csv=p=0:s=x", src], capture_output=True, text=True).stdout.strip().split("x"))
font, k = spec.get("font", "Hiragino Sans GB"), W / 1080
ts = lambda t: f"0:{int(t // 60):02d}:{t % 60:05.2f}"
ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{round(62 * k)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{2.8 * k:.1f},{2.8 * k:.1f},2,40,40,{round(H * 0.195)},1
Style: Pop,{font},{round(170 * k)},&H0000D6FF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{9 * k:.1f},{4 * k:.1f},5,30,30,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
for a, b, t in spec.get("pops", []):
    ass += (f"Dialogue: 2,{ts(a)},{ts(b)},Pop,,0,0,0,,{{\\pos({W // 2},{round(H * 0.66)})\\fad(60,120)"
            f"\\fscx60\\fscy60\\t(0,160,\\fscx108\\fscy108)\\t(160,260,\\fscx100\\fscy100)}}{t}\n")
for a, b, t in spec["lines"]:
    ass += f"Dialogue: 0,{ts(a)},{ts(b)},Default,,0,0,0,,{t}\n"
open(out + ".ass", "w").write(ass)

r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                   capture_output=True, text=True)
m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
ln = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
      f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
vf = f"subtitles={out}.ass" + (f":fontsdir={fontsdir}" if fontsdir else "")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf, "-af", ln + ",aresample=48000", "-c:v", "libx264", "-crf", "16",
                "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out], check=True)
print("done", out)
