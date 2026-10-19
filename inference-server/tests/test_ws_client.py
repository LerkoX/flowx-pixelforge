"""flowx_client 的 WS 等待路径真测（纯 stdlib，内嵌迷你 WS/HTTP 服务器）。

不起 fastapi：服务端用 socket + 线程实现 RFC 6455 握手与发帧（服务端帧不掩码），
同时提供 /jobs/{id} REST 轮询端点（计数 ⇒ 可断言「WS 路径轮询次数」）。

覆盖（含负例自检）：
1. 事件驱动到 done：snapshot → progress×2 → done；REST 只在终态确认打 1 次；
2. 握手被拒（HTTP 200 而非 101）→ 自动回退轮询，任务正常完成；
3. WS 中途断开 → 回退轮询完成（**负例**：直接调 _wait_job_ws 必须抛 _WsFallback，
   证明是回退救了它而不是别的）；
4. 协议级 ping → 客户端回 pong（服务端记录到 pong 帧）；
5. FLOWX_WS=0 → 完全不尝试 WS 连接。

运行：python3 tests/test_ws_client.py
"""
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "nodes", "_common"))

import flowx_client as fc  # noqa: E402

FAILED = []
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def check(label, cond):
    print(("✓" if cond else "✗"), label)
    if not cond:
        FAILED.append(label)


def _frame(op, payload=b""):
    """服务端帧（不掩码）。"""
    h = bytearray([0x80 | op])
    n = len(payload)
    if n < 126:
        h.append(n)
    elif n < 65536:
        h += bytes([126]) + struct.pack(">H", n)
    else:
        h += bytes([127]) + struct.pack(">Q", n)
    return bytes(h) + payload


def _read_exact(conn, n):
    d = b""
    while len(d) < n:
        c = conn.recv(n - len(d))
        if not c:
            raise ConnectionError("closed")
        d += c
    return d


class MiniServer:
    """script = list of ("ws", dict|bytes|("ping",payload)|"close") 或 ("job", dict)。
    WS 升级后按序发 script 里的 ws 项；REST /jobs/{id} 依次返回 job 项（用完后
    重复最后一项）。统计：ws_handshakes / poll_hits / pongs。"""

    def __init__(self, ws_script, job_views, ws_ok=True):
        self.ws_script = list(ws_script)
        self.job_views = list(job_views)
        self.ws_ok = ws_ok
        self.ws_handshakes = 0
        self.poll_hits = 0
        self.pongs = 0
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self._stop = True
        self.srv.close()

    def _accept(self):
        while not self._stop:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn):
        try:
            req = b""
            while b"\r\n\r\n" not in req:
                c = conn.recv(4096)
                if not c:
                    conn.close()
                    return
                req += c
            head = req.decode(errors="replace")
            if "upgrade: websocket" in head.lower():
                self.ws_handshakes += 1
                if not self.ws_ok:
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                                 b"\r\n{}")
                    conn.close()
                    return
                key = [l.split(":", 1)[1].strip() for l in head.split("\r\n")
                       if l.lower().startswith("sec-websocket-key")][0]
                accept = base64.b64encode(
                    hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
                conn.sendall((f"HTTP/1.1 101 Switching Protocols\r\n"
                              f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
                self._ws_loop(conn)
                return
            # REST: GET /jobs/{id}
            self.poll_hits += 1
            v = self.job_views.pop(0) if len(self.job_views) > 1 \
                else self.job_views[0]
            body = json.dumps(v).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            conn.close()
        except Exception:
            try:
                conn.close()
            except OSError:
                pass

    def _ws_loop(self, conn):
        conn.settimeout(8)
        for item in self.ws_script:
            if item == "close":
                conn.close()
                return
            if isinstance(item, tuple) and item[0] == "ping":
                conn.sendall(_frame(0x9, item[1]))
                # 等客户端 pong（掩码帧）
                hdr = _read_exact(conn, 2)
                op = hdr[0] & 0x0F
                ln = hdr[1] & 0x7F
                mask = _read_exact(conn, 4)
                _read_exact(conn, ln)
                if op == 0xA:
                    self.pongs += 1
                continue
            if isinstance(item, (bytes, bytearray)):
                conn.sendall(bytes(item))
                continue
            conn.sendall(_frame(0x1, json.dumps(item).encode()))
        # 剧本演完后保持连接（等客户端读超时或自己走）
        try:
            while True:
                if not conn.recv(4096):
                    return
        except OSError:
            return


def base_env():
    fc._WS_DISABLED = False  # 每个用例前复位（用例 5 会改）


# ---------- 1. 事件驱动到 done ----------
def test_ws_happy_path():
    base_env()
    srv = MiniServer(
        ws_script=[{"type": "snapshot",
                    "job": {"status": "running",
                            "progress": {"current": 0, "total": 3}}},
                   {"type": "progress", "current": 1, "total": 3},
                   {"type": "progress", "current": 3, "total": 3},
                   {"type": "done"}],
        job_views=[{"status": "done", "result": {"image": "obj-9"}}])
    seen = []
    r = fc.wait_job(srv.url(), "jobA", tok=None, timeout=30, poll=0.1,
                    log=None, on_poll=lambda v: seen.append(dict(v)))
    srv.close()
    check("WS 拿到 result", r == {"image": "obj-9"})
    check("WS 路径 REST 仅终态确认 1 次（零轮询）", srv.poll_hits == 1)
    check("on_poll 收到进度视图", any((v.get("progress") or {}).get("current") == 1
                                     for v in seen))


# ---------- 2. 握手被拒 → 回退轮询 ----------
def test_fallback_on_handshake_refused():
    base_env()
    srv = MiniServer(ws_script=[], ws_ok=False,
                     job_views=[{"status": "running",
                                 "progress": {"current": 0, "total": 1}},
                                {"status": "done", "result": {"ok": 1}}])
    logs = []
    r = fc.wait_job(srv.url(), "jobB", timeout=30, poll=0.1, log=logs.append)
    srv.close()
    check("回退轮询拿到结果", r == {"ok": 1})
    check("尝试过 WS（握手 1 次）", srv.ws_handshakes == 1)
    check("回退后走轮询", srv.poll_hits >= 2)
    check("日志告知回退", any("回退轮询" in l for l in logs))


# ---------- 3. 中途断开 → 回退；负例：无回退必败 ----------
def test_fallback_on_midstream_break():
    base_env()
    srv = MiniServer(
        ws_script=[{"type": "progress", "current": 1, "total": 5}, "close"],
        job_views=[{"status": "done", "result": {"ok": 2}}])
    r = fc.wait_job(srv.url(), "jobC", timeout=30, poll=0.1, log=None)
    srv.close()
    check("中途断开后回退轮询完成", r == {"ok": 2})


def test_negative_no_fallback_must_fail():
    base_env()
    srv = MiniServer(ws_script=[{"type": "progress", "current": 1, "total": 5},
                                "close"],
                     job_views=[{"status": "done", "result": {}}])
    try:
        fc._wait_job_ws(srv.url(), "jobD", None, 30, None, None)
        check("负例：直接走 WS 遇到断开必须抛 _WsFallback", False)
    except fc._WsFallback:
        check("负例：直接走 WS 遇到断开必须抛 _WsFallback", True)
    except Exception as e:
        check(f"负例：抛错类型应为 _WsFallback（实际 {type(e).__name__}: {e}）",
              False)
    srv.close()


# ---------- 4. 协议 ping → 客户端回 pong ----------
def test_ping_pong():
    base_env()
    srv = MiniServer(
        ws_script=[("ping", b"hb"), {"type": "done"}],
        job_views=[{"status": "done", "result": {"ok": 3}}])
    r = fc.wait_job(srv.url(), "jobE", timeout=30, poll=0.1, log=None)
    srv.close()
    check("pong 已回（服务端收到 0xA 帧）", srv.pongs == 1 and r == {"ok": 3})


# ---------- 5. FLOWX_WS=0 禁用 ----------
def test_ws_disabled_env():
    base_env()
    fc._WS_DISABLED = True
    srv = MiniServer(ws_script=[],
                     job_views=[{"status": "done", "result": {"ok": 4}}])
    r = fc.wait_job(srv.url(), "jobF", timeout=30, poll=0.1, log=None)
    srv.close()
    fc._WS_DISABLED = False
    check("FLOWX_WS=0 完全不尝试 WS", srv.ws_handshakes == 0 and r == {"ok": 4})


def main():
    test_ws_happy_path()
    test_fallback_on_handshake_refused()
    test_fallback_on_midstream_break()
    test_negative_no_fallback_must_fail()
    test_ping_pong()
    test_ws_disabled_env()
    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
