"""按 assets.json 批量出资产图（H3 Image，8188）。每张一个种子，出完改成中文名。用法：make_assets.py assets.json"""
import json, os, sys, subprocess, glob, shutil
cfg = json.load(open(sys.argv[1])); out = cfg["out"]; os.makedirs(out, exist_ok=True)
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # H3-Video/scripts
env = dict(os.environ)  # COMFY_URL 指向跑 H3 Image 的 ComfyUI
for j in cfg["jobs"]:
    if glob.glob(os.path.join(out, j["name"] + ".png")): continue
    tmp = os.path.join(out, "_tmp"); os.makedirs(tmp, exist_ok=True)
    cmd = ["python3", f"{S}/h3_image.py", j["prompt"]] + sum([["--img", i] for i in j["imgs"]], []) + \
          ["--size", j.get("size", "1728x2368"), "--seeds", str(j["seed"]), "--prefix", f"assets/{j['name']}", "--wait", tmp]
    r = subprocess.run(cmd, cwd=S, env=env, capture_output=True, text=True)
    files = sorted(glob.glob(os.path.join(tmp, "*.png")), key=os.path.getmtime)
    if files: shutil.move(files[-1], os.path.join(out, j["name"] + ".png")); print("ok", j["name"], flush=True)
    else: print("FAIL", j["name"], r.stdout[-400:], r.stderr[-400:], flush=True)
