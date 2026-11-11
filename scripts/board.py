"""写工作台上的“Claude 在做什么”。dashboard.py 页面会显示 ~/.h3board/board.json。

用法：
  python3 board.py start "第 6 集 狗狗段子"            # 开新任务，清空步骤
  python3 board.py plan "写台词" "出首帧" "首帧给你确认" "生成视频" "配字幕"
  python3 board.py step "出首帧" running "Qwen-Image 4 张，约 6 分钟"   # 状态：pending/running/done/wait/failed
  python3 board.py note "首帧第 2 张眼神不对，重出"
  python3 board.py file /path/to/frame.png "首帧候选 2"   # 本地文件也能在页面上看
  python3 board.py feedback                            # 打印页面上的新留言（读过的会标记）
"""
import json, os, sys, time

DIR = os.path.expanduser("~/.h3board")
BOARD = os.path.join(DIR, "board.json")
FEEDBACK = os.path.join(DIR, "feedback.jsonl")
SEEN = os.path.join(DIR, "feedback.seen")


def load():
    try:
        with open(BOARD, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"title": "", "steps": [], "notes": [], "files": []}


def save(b):
    os.makedirs(DIR, exist_ok=True)
    b["updated"] = time.time()
    tmp = BOARD + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)
    os.replace(tmp, BOARD)


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, args = argv[0], argv[1:]
    b = load()
    now = time.time()
    if cmd == "start":
        b = {"title": args[0] if args else "", "started": now, "steps": [], "notes": [], "files": []}
    elif cmd == "plan":
        old = {s["name"]: s for s in b.get("steps", [])}
        b["steps"] = [old.get(n, {"name": n, "status": "pending"}) for n in args]
    elif cmd == "step":
        name, status = args[0], (args[1] if len(args) > 1 else "running")
        s = next((x for x in b["steps"] if x["name"] == name), None)
        if not s:
            s = {"name": name}
            b["steps"].append(s)
        s["status"] = status
        if len(args) > 2:
            s["detail"] = args[2]
        s.setdefault("since", now)
        if status == "running":
            s["since"] = now
        if status in ("done", "failed"):
            s["end"] = now
    elif cmd == "note":
        b.setdefault("notes", []).append({"time": now, "text": " ".join(args)})
        b["notes"] = b["notes"][-50:]
    elif cmd == "file":
        p = os.path.abspath(args[0])
        b.setdefault("files", []).append({"path": p, "label": args[1] if len(args) > 1 else os.path.basename(p), "time": now})
        b["files"] = b["files"][-40:]
    elif cmd == "feedback":
        seen = int(open(SEEN).read() or 0) if os.path.exists(SEEN) else 0
        rows = open(FEEDBACK, encoding="utf-8").read().splitlines() if os.path.exists(FEEDBACK) else []
        for r in rows[seen:]:
            x = json.loads(r)
            print(x["time"], x.get("ref", ""), x["text"])
        os.makedirs(DIR, exist_ok=True)
        open(SEEN, "w").write(str(len(rows)))
        return
    else:
        raise SystemExit(__doc__)
    save(b)


if __name__ == "__main__":
    main(sys.argv[1:])
