"""监听 ComfyUI 进度 N 秒，打印节点执行事件和最后一次步数进度。

ComfyUI 只把进度发给提交任务的那个 client_id，所以这里用 comfy.CLIENT_ID。
用别的 id 监听什么都收不到，会误以为任务卡死。

用法：python3 ws_progress.py [秒数，默认 60]
"""
import base64, json, os, socket, struct, sys, time, urllib.parse
import comfy

u = urllib.parse.urlparse(comfy.URL)
host, port = u.hostname, u.port or 80
s = socket.create_connection((host, port), timeout=10)
key = base64.b64encode(os.urandom(16)).decode()
s.send(f"GET /ws?clientId={comfy.CLIENT_ID} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
       f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
buf = b""
while b"\r\n\r\n" not in buf:
    buf += s.recv(4096)
buf = buf.split(b"\r\n\r\n", 1)[1]
t, last = time.time(), None
s.settimeout(3)
while time.time() - t < float(sys.argv[1] if len(sys.argv) > 1 else 60):
    try:
        d = s.recv(65536)
        if not d:
            break
        buf += d
    except socket.timeout:
        pass
    while len(buf) >= 2:
        b1, n, i = buf[0], buf[1] & 127, 2
        if n == 126:
            n, i = struct.unpack(">H", buf[2:4])[0], 4
        elif n == 127:
            n, i = struct.unpack(">Q", buf[2:10])[0], 10
        if len(buf) < i + n:
            break
        payload, buf = buf[i:i + n], buf[i + n:]
        if b1 & 15 != 1:
            continue
        try:
            m = json.loads(payload)
        except ValueError:
            continue
        d = m.get("data", {})
        if m["type"] == "progress":
            last = (round(time.time() - t), d.get("value"), d.get("max"), d.get("node"))
        elif m["type"] in ("executing", "executed", "execution_start", "execution_error", "execution_interrupted"):
            print(round(time.time() - t), m["type"], str(d)[:100])
print("最后进度（秒, 当前步, 总步数, 节点）", last)
