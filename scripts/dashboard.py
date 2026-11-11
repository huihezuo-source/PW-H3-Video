"""本地工作台页面：在浏览器里实时看 ComfyUI 在跑什么、跑到第几步、出了哪些片，以及 Claude 现在做到哪一步。

只用 Python 标准库。启动后浏览器打开 http://localhost:8765 。
  python3 dashboard.py [--port 8765]

看哪几台机器：默认是家里电脑的 8188 和 8189（走 Tailscale）。要加别的机器（比如 01），
在本文件同目录建 dashboard_machines.json：
  [{"name": "家里 8188", "url": "http://100.122.217.58:8188"}, {"name": "01", "url": "http://100.x.x.x:8188"}]

进度怎么拿到：ComfyUI 只把步数进度发给提交任务时的 client_id。本页面在后台用 mac-claude 和 h3-video
两个 id 各开一条 websocket（脚本提交任务都用这两个 id 之一）。同一个 id 只能有一个监听者，
所以页面开着时不要再跑 ws_progress.py，否则页面的进度会停住（页面 60 秒收不到进度会自动重连抢回来）。

Claude 的工作进度：脚本或 Claude 用 board.py 写 ~/.h3board/board.json，页面显示出来。
页面上的留言写到 ~/.h3board/feedback.jsonl，Claude 每一步开始前读。
"""
import argparse, base64, json, os, socket, struct, threading, time, urllib.parse, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_DIR = os.path.expanduser("~/.h3board")
BOARD = os.path.join(BOARD_DIR, "board.json")
FEEDBACK = os.path.join(BOARD_DIR, "feedback.jsonl")
CLIENT_IDS = ["mac-claude", "h3-video"]
DEFAULT_MACHINES = [
    {"name": "家里 8188", "url": "http://100.122.217.58:8188"},
    {"name": "家里 8189（H3）", "url": "http://100.122.217.58:8189"},
]


def load_machines():
    p = os.environ.get("H3_BOARD_MACHINES") or os.path.join(HERE, "dashboard_machines.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_MACHINES


def get_json(url, timeout=6):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post_json(url, body, timeout=10):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read()
    return json.loads(txt) if txt.strip() else {}


def describe(prompt):
    """从 API 格式工作流里挑出人能看懂的信息：提示词、分辨率、帧数、步数、种子、输出名、用到的模型。"""
    texts, info, models, kinds = [], {}, [], []
    for nid, node in (prompt or {}).items():
        ct, inp = node.get("class_type", ""), node.get("inputs", {})
        kinds.append(ct)
        for k, v in inp.items():
            if isinstance(v, str):
                if k in ("prompt", "text", "positive", "text_g", "prompt_text") and len(v) > 8:
                    texts.append(v)
                elif k.endswith("_name") and v.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".gguf")):
                    models.append(v)
                elif k == "filename_prefix":
                    info["输出"] = v
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in ("width", "height", "length", "steps", "seed", "noise_seed", "fps", "cfg"):
                    info.setdefault(k, v)
    title = info.get("输出") or (kinds[0] if kinds else "任务")
    size = f'{info["width"]}×{info["height"]}' if "width" in info and "height" in info else ""
    return {
        "title": title,
        "size": size,
        "frames": info.get("length"),
        "steps": info.get("steps"),
        "seed": info.get("seed", info.get("noise_seed")),
        "text": max(texts, key=len) if texts else "",
        "models": sorted(set(models)),
        "nodes": {nid: n.get("class_type", "") for nid, n in (prompt or {}).items()},
    }


class Listener(threading.Thread):
    """用一个 client_id 连 ComfyUI 的 websocket，记下每个任务的实时进度。"""

    def __init__(self, machine, cid, live):
        super().__init__(daemon=True)
        self.m, self.cid, self.live = machine, cid, live
        self.last_msg = 0
        self.kick = threading.Event()

    def run(self):
        while True:
            try:
                self.loop()
            except Exception:
                pass
            time.sleep(5)

    def loop(self):
        u = urllib.parse.urlparse(self.m["url"])
        host, port = u.hostname, u.port or 80
        s = socket.create_connection((host, port), timeout=8)
        key = base64.b64encode(os.urandom(16)).decode()
        s.send(f"GET /ws?clientId={self.cid} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = s.recv(4096)
            if not d:
                raise ConnectionError("closed")
            buf += d
        head, buf = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError("no upgrade")
        s.settimeout(2)
        self.last_msg = time.time()
        self.kick.clear()
        while not self.kick.is_set():
            try:
                d = s.recv(1 << 16)
                if not d:
                    raise ConnectionError("closed")
                buf += d
            except socket.timeout:
                continue
            while len(buf) >= 2:
                b1, n, i = buf[0], buf[1] & 127, 2
                if n == 126:
                    if len(buf) < 4:
                        break
                    n, i = struct.unpack(">H", buf[2:4])[0], 4
                elif n == 127:
                    if len(buf) < 10:
                        break
                    n, i = struct.unpack(">Q", buf[2:10])[0], 10
                if len(buf) < i + n:
                    break
                payload, buf = buf[i:i + n], buf[i + n:]
                op = b1 & 15
                if op == 8:
                    raise ConnectionError("server closed")
                if op == 9:   # ping → pong（客户端发出的帧必须加掩码）
                    s.send(bytes([0x8A, 0x80 | len(payload[:125])]) + b"\0\0\0\0" + payload[:125])
                    continue
                if op != 1:
                    continue   # 二进制帧是预览图，先不管
                try:
                    self.handle(json.loads(payload))
                except ValueError:
                    pass
        s.close()

    def handle(self, m):
        t, d = m.get("type"), m.get("data", {}) or {}
        pid = d.get("prompt_id")
        now = time.time()
        if t in ("progress", "executing", "execution_start", "execution_cached", "progress_state"):
            self.last_msg = now
        if not pid:
            return
        with self.live["lock"]:
            p = self.live["prompts"].setdefault(pid, {"start": now, "steps": []})
            p["machine"], p["updated"] = self.m["name"], now
            if t == "execution_start":
                p.update(start=now, node=None, value=None, max=None, steps=[])
            elif t == "executing":
                if d.get("node") is None:
                    p["done"] = now
                elif d["node"] != p.get("node"):
                    p.update(node=d["node"], node_start=now, value=None, max=None, steps=[])
            elif t == "progress":
                if d.get("node") and d["node"] != p.get("node"):
                    p.update(node=d["node"], node_start=now, steps=[])
                if d.get("value") != p.get("value"):
                    p["steps"] = (p.get("steps") or [])[-40:] + [now]
                p["value"], p["max"] = d.get("value"), d.get("max")
            elif t == "execution_error":
                p["error"] = (d.get("node_type") or "") + "：" + str(d.get("exception_message", ""))[:300]
            elif t == "execution_interrupted":
                p["error"] = "已中断"


class State:
    def __init__(self):
        self.machines = load_machines()
        self.live = {"lock": threading.Lock(), "prompts": {}}
        self.listeners = {}
        self.cache = {}
        for m in self.machines:
            self.listeners[m["name"]] = [Listener(m, cid, self.live) for cid in CLIENT_IDS]
            for l in self.listeners[m["name"]]:
                l.start()
            threading.Thread(target=self.poll, args=(m,), daemon=True).start()

    def machine(self, name):
        return next(m for m in self.machines if m["name"] == name)

    def poll(self, m):
        """每 2 秒拉一次队列、显存和最近历史，放进缓存，页面读缓存，不会被慢接口卡住。"""
        n = 0
        while True:
            c = {"name": m["name"], "url": m["url"], "checked": time.time()}
            try:
                st = get_json(m["url"] + "/system_stats", timeout=8)
                c["online"] = True
                dev = (st.get("devices") or [{}])[0]
                c["gpu"] = dev.get("name", "").replace("cuda:0 ", "")
                c["vram_total"], c["vram_free"] = dev.get("vram_total"), dev.get("vram_free")
                c["comfy_version"] = st.get("system", {}).get("comfyui_version")
                q = get_json(m["url"] + "/queue", timeout=8)
                c["running"] = [self.item(x, m) for x in q.get("queue_running", [])]
                c["pending"] = sorted((self.item(x, m) for x in q.get("queue_pending", [])), key=lambda x: x["number"])
                if n % 3 == 0 or "history" not in self.cache.get(m["name"], {}):
                    h = get_json(m["url"] + "/history?max_items=24", timeout=15)
                    c["history"] = self.history(h, m)
                else:
                    c["history"] = self.cache[m["name"]]["history"]
                # 有任务在跑却 60 秒没收到任何进度：可能被别的监听者抢了 id，重连抢回来
                if c["running"]:
                    for l in self.listeners[m["name"]]:
                        if time.time() - l.last_msg > 60:
                            l.kick.set()
                            l.last_msg = time.time()
            except Exception as e:
                c["online"] = False
                c["error"] = str(e)[:120]
                old = self.cache.get(m["name"], {})
                c["history"] = old.get("history", [])
            self.cache[m["name"]] = c
            n += 1
            time.sleep(2)

    def item(self, x, m):
        number, pid, prompt = x[0], x[1], x[2]
        extra = x[3] if len(x) > 3 else {}
        d = describe(prompt)
        with self.live["lock"]:
            lp = dict(self.live["prompts"].get(pid, {}))
        out = {"number": number, "id": pid, "client": extra.get("client_id"), **d}
        if lp:
            node = lp.get("node")
            out["live"] = {
                "start": lp.get("start"), "node": node, "node_type": d["nodes"].get(str(node), ""),
                "value": lp.get("value"), "max": lp.get("max"), "node_start": lp.get("node_start"),
                "sec_per_step": self.rate(lp.get("steps") or []), "updated": lp.get("updated"),
            }
        return out

    @staticmethod
    def rate(ts):
        if len(ts) < 3:
            return None
        gaps = [b - a for a, b in zip(ts[:-1], ts[1:])][-10:]
        return sum(gaps) / len(gaps)

    def history(self, h, m):
        rows = []
        for pid, rec in h.items():
            pr = rec.get("prompt", [])
            d = describe(pr[2] if len(pr) > 2 else {})
            st = rec.get("status", {})
            ts = {mm[0]: mm[1].get("timestamp") for mm in st.get("messages", []) if isinstance(mm[1], dict)}
            err = next((mm[1] for mm in st.get("messages", []) if mm[0] == "execution_error"), None)
            end = ts.get("execution_success") or ts.get("execution_error") or ts.get("execution_interrupted")
            files = []
            for o in rec.get("outputs", {}).values():
                for k in ("images", "video", "gifs", "audio", "videos"):
                    for f in o.get(k, []) or []:
                        if isinstance(f, dict) and f.get("filename"):
                            files.append({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                          "type": f.get("type", "output"), "kind": kind_of(f["filename"], k)})
            rows.append({
                "id": pid, "number": pr[0] if pr else 0, **d,
                "status": st.get("status_str") or ("success" if st.get("completed") else "?"),
                "start": (ts.get("execution_start") or 0) / 1000, "end": (end or 0) / 1000,
                "error": (err.get("node_type", "") + "：" + str(err.get("exception_message", ""))[:300]) if err else "",
                "files": files,
            })
        rows.sort(key=lambda r: r["number"], reverse=True)
        return rows

    def snapshot(self):
        board, fb = {}, []
        try:
            with open(BOARD, encoding="utf-8") as f:
                board = json.load(f)
        except Exception:
            pass
        try:
            with open(FEEDBACK, encoding="utf-8") as f:
                fb = [json.loads(x) for x in f if x.strip()][-20:]
        except Exception:
            pass
        return {"now": time.time(), "machines": [self.cache.get(m["name"], {"name": m["name"], "url": m["url"]})
                                                  for m in self.machines], "board": board, "feedback": fb}

    def workflow(self, name, pid):
        m = self.machine(name)
        h = get_json(m["url"] + "/history/" + urllib.parse.quote(pid), timeout=15)
        if pid in h:
            return h[pid]["prompt"][2]
        q = get_json(m["url"] + "/queue", timeout=15)
        for x in q.get("queue_running", []) + q.get("queue_pending", []):
            if x[1] == pid:
                return x[2]
        raise KeyError(pid)


def kind_of(name, key):
    n = name.lower()
    if n.endswith((".mp4", ".webm", ".mov", ".mkv")):
        return "video"
    if n.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")) or key == "audio":
        return "audio"
    return "image"


STATE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self.send(200, STATE.snapshot())
            if u.path == "/api/view":
                return self.proxy_view(q)
            if u.path == "/api/local":   # 只给 board.json 里登记过的本地文件
                allowed = {f["path"] for f in STATE.snapshot()["board"].get("files", [])}
                p = q.get("path", "")
                if p not in allowed or not os.path.isfile(p):
                    return self.send(404, {"error": "not listed"})
                import mimetypes
                with open(p, "rb") as f:
                    return self.send(200, f.read(), mimetypes.guess_type(p)[0] or "application/octet-stream")
            if u.path == "/api/workflow":
                wf = STATE.workflow(q["m"], q["id"])
                return self.send(200, json.dumps(wf, ensure_ascii=False, indent=1), "application/json; charset=utf-8",
                                 {"Content-Disposition": f'attachment; filename="workflow_{q["id"][:8]}.json"'})
            self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(500, {"error": str(e)[:300]})

    def proxy_view(self, q):
        m = STATE.machine(q.pop("m"))
        url = m["url"] + "/view?" + urllib.parse.urlencode(q)
        hdr = {}
        if self.headers.get("Range"):
            hdr["Range"] = self.headers["Range"]
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60)
        except urllib.error.HTTPError as e:
            r = e
        self.send_response(r.status if hasattr(r, "status") else r.code)
        for k in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            if r.headers.get(k):
                self.send_header(k, r.headers[k])
        self.send_header("Cache-Control", "max-age=600")
        self.end_headers()
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            try:
                self.wfile.write(b)
            except (BrokenPipeError, ConnectionResetError):
                break

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if u.path == "/api/feedback":
                os.makedirs(BOARD_DIR, exist_ok=True)
                with open(FEEDBACK, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": time.strftime("%m-%d %H:%M:%S"), "text": body.get("text", "")[:2000],
                                        "ref": body.get("ref", "")}, ensure_ascii=False) + "\n")
                return self.send(200, {"ok": True})
            m = STATE.machine(body["m"])
            if u.path == "/api/interrupt":   # 只停正在跑的这一条
                post_json(m["url"] + "/interrupt", {"prompt_id": body.get("id")} if body.get("id") else {})
                return self.send(200, {"ok": True})
            if u.path == "/api/delete":      # 从排队里删掉，不影响正在跑的
                post_json(m["url"] + "/queue", {"delete": [body["id"]]})
                return self.send(200, {"ok": True})
            self.send(404, {"error": "not found"})
        except Exception as e:
            self.send(500, {"error": str(e)[:300]})


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    STATE = State()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"工作台已启动：http://localhost:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
