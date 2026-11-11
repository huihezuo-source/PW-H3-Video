"""等一个任务结束，下载全部输出（图片、音频、视频）。

用法：python3 wait_download.py <prompt_id> <out_dir> [超时秒数，默认 3600]
"""
import sys, time
import comfy

pid, out = sys.argv[1], sys.argv[2]
t = time.time()
rec = comfy.wait(pid, timeout=int(sys.argv[3]) if len(sys.argv) > 3 else 3600)
print(rec["status"].get("status_str"), f"{time.time() - t:.0f}s")
for f in comfy.download(rec, out):
    print(f)
