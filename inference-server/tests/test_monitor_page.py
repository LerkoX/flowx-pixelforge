"""监控页（/monitor + app/static/monitor.html）静态约束测试。

本机测试环境无 fastapi/torch（项目测试惯例：stub 重依赖、纯 stdlib 可跑），
不起 TestClient，改为断言：
- 静态页存在且引用了全部数据/操作端点（漏配端点 = 页面某区块永远不更新）
- main.py 注册了 /monitor 与 /gpu 路由
- /monitor 不设鉴权（页面无数据），/gpu 走 auth（数据端点统一 Bearer 闸门）

运行：python3 tests/test_monitor_page.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
HTML = os.path.join(ROOT, "app", "static", "monitor.html")
MAIN = os.path.join(ROOT, "app", "main.py")


def test_monitor_html_exists_and_references_all_endpoints():
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    # 数据源端点
    for ep in ["/health", "/gpu", "/leases", "/jobs", "/metrics/history"]:
        assert f'"{ep}"' in html or f'"{ep}?' in html, f"monitor.html 未引用数据端点 {ep}"
    # 运维操作端点
    for ep in ["/gc", "/model/unload", "/interrupt"]:
        assert ep in html, f"monitor.html 未引用操作端点 {ep}"
    # token 存 localStorage（页面无鉴权，凭据不进服务端）
    assert "localStorage" in html and "Authorization" in html


def test_main_registers_monitor_and_gpu_routes():
    with open(MAIN, encoding="utf-8") as f:
        src = f.read()
    assert '@app.get("/monitor")' in src
    assert '@app.get("/gpu"' in src
    assert '@app.get("/metrics/history"' in src
    # 数据端点必须过 auth；/monitor 页面本身无数据不设闸门
    m = re.search(r'@app\.get\("/gpu"[^)]*\)', src)
    assert m and "auth" in m.group(0)
    m = re.search(r'@app\.get\("/metrics/history"[^)]*\)', src)
    assert m and "auth" in m.group(0)
    m = re.search(r'@app\.get\("/monitor"[^)]*\)', src)
    assert m and "auth" not in m.group(0)
    # 静态目录按 __file__ 定位（uvicorn 工作目录无关）
    assert "os.path.dirname(os.path.abspath(__file__))" in src


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
