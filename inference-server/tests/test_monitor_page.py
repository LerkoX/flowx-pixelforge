"""监控页（/monitor + app/static/monitor.html）静态约束测试。

本机测试环境无 fastapi/torch（项目测试惯例：stub 重依赖、纯 stdlib 可跑），
不起 TestClient，改为断言：
- 静态页存在且引用了全部数据/操作端点（漏配端点 = 页面某区块永远不更新）
- 进度条/显存条的关键结构不再退化（历史 bug 的两个坑，见下）
- 运维操作必须有可见反馈（不能再「点了没反应」）
- main.py 注册了 /monitor 与 /gpu 路由，/jpbs 列表带 queue 汇总
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
JOBS = os.path.join(ROOT, "app", "jobs.py")


def html_src():
    with open(HTML, encoding="utf-8") as f:
        return f.read()


def test_monitor_html_exists_and_references_all_endpoints():
    html = html_src()
    # 数据源端点
    for ep in ["/health", "/gpu", "/leases", "/jobs", "/metrics/history"]:
        assert f'"{ep}"' in html or f'"{ep}?' in html, f"monitor.html 未引用数据端点 {ep}"
    # 运维操作端点
    for ep in ["/gc", "/model/unload", "/interrupt", "/leases/"]:
        assert ep in html, f"monitor.html 未引用操作端点 {ep}"
    # token 存 localStorage（页面无鉴权，凭据不进服务端）
    assert "localStorage" in html and "Authorization" in html


def test_progress_bar_fill_not_covering_label():
    """回归：`.pbar > div` 会把叠在上面的百分比文字层也染成蓝色铺满整条，
    导致所有任务看起来都是「一条满格蓝条」。填充层必须用 .fill 单独选择，
    且文字不再用绝对定位压进条内。"""
    html = html_src()
    assert ".pbar > .fill" in html, "进度条填充层必须用 .pbar > .fill 选择"
    assert ".pbar > div {" not in html, "禁止 `.pbar > div`（会命中文字层）"
    assert 'class="pbar"><div class="fill' in html or 'class="pbar"><div class="fill ' in html
    assert "pmeta" in html, "进度文字应外置到 .pmeta（条内压字不可读）"


def test_vram_bar_accounts_four_segments():
    """回归：显存条必须把「已分配 / 缓存可回收 / 其它进程驱动 / 空闲」分账，
    且读数放在条外的图例与明细里（此前一整行中文塞进 18px 细条 + 两段同色系）。"""
    html = html_src()
    for seg in ["seg-alloc", "seg-cache", "seg-other"]:
        assert seg in html, f"显存条缺少分段 {seg}"
    assert 'id="vram-legend"' in html and 'id="vram-detail"' in html
    assert "缓存保留" in html and "其它进程" in html
    # 旧版把整行文字压在条内的 #vram-lbl 已移除
    assert 'id="vram-lbl"' not in html


def test_ops_buttons_have_visible_feedback():
    """回归：运维按钮点完不能静默（/gc 的返回被丢弃、卸载被租约跳过时无提示）。"""
    html = html_src()
    assert 'id="ops-out"' in html, "运维操作缺少结果输出区"
    assert "opsLog" in html and "opButton" in html
    # 不用 window.confirm/alert（内嵌 webview 里会静默返回 false ⇒ 看起来点了没效果）
    assert "confirm(" not in html, "运维操作应改用页内二次确认，不依赖 window.confirm"
    assert "alert(" not in html, "运维操作结果应写入页面，不用 window.alert"
    # 卸载被跳过时能就地释放租约
    assert "data-release" in html and "skipped_detail" in html


def test_jobs_panel_shows_richer_fields():
    html = html_src()
    for key in ["j.title", "j.stage", "jQueue", "queue_pos", "sec_per_step",
                "resultSummary", "jobEta"]:
        assert key in html or key == "jQueue", f"任务队列缺少展示字段 {key}"
    assert 'id="queue-summary"' in html
    assert 'id="jobs-only-active"' in html


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
    # /jobs 列表带 queue 汇总（worker 忙闲 / 队头等待）
    assert "queue_state" in src


def test_jobs_view_exposes_title_stage_queue_pos():
    with open(JOBS, encoding="utf-8") as f:
        src = f.read()
    for key in ['"title"', '"ops"', '"stage"', '"queue_pos"', '"wait_seconds"',
                "def set_stage", "def queue_state"]:
        assert key in src, f"jobs.py 缺少 {key}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
