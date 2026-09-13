"""图执行引擎单元测试（无需 GPU/torch，用假算子验证拓扑排序、缓存、类型校验）。
运行：python3 -m tests.test_engine  或  python3 tests/test_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine import run_graph, topo_sort
from app.object_store import ObjectStore
from app.registry import Registry


def build_registry():
    reg = Registry()

    @reg.register("make", inputs={"value": "INT"}, outputs={"obj": "LATENT"})
    def _(value):
        calls.append("make")
        return {"obj": value * 10}

    @reg.register("double", inputs={"x": "LATENT"}, outputs={"y": "LATENT"})
    def _(x):
        calls.append("double")
        return {"y": x * 2}

    @reg.register("merge", inputs={"a": "LATENT", "b": "LATENT", "k": "INT"},
                  outputs={"out": "LATENT", "total": "INT"})
    def _(a, b, k=1):
        calls.append("merge")
        return {"out": a + b, "total": (a + b) * k}

    return reg


GRAPH = {
    "nodes": {
        "1": {"op": "make", "inputs": {"value": 3}},
        "2": {"op": "double", "inputs": {"x": ["1", "obj"]}},
        "3": {"op": "make", "inputs": {"value": 5}},
        "4": {"op": "merge", "inputs": {"a": ["2", "y"], "b": ["3", "obj"], "k": 2}},
    }
}


def main():
    global calls

    # --- 拓扑排序 ---
    order = topo_sort(GRAPH["nodes"])
    assert order.index("1") < order.index("2") < order.index("4")
    assert order.index("3") < order.index("4")
    print("topo_sort OK:", order)

    # --- 首次执行 ---
    calls = []
    store, reg, cache = ObjectStore(), build_registry(), {}
    r1 = run_graph(store, reg, GRAPH, cache)
    # make(3)=30 -> double=60; make(5)=50; merge out=110, total=220
    m4 = r1["nodes"]["4"]
    assert m4["total"] == {"value": 220, "type": "INT"}, m4
    assert store.get(m4["out"]["id"], "LATENT")["data"] == 110
    assert sorted(calls) == ["double", "make", "make", "merge"], calls
    assert r1["cached"] == []
    print("first run OK:", r1["nodes"]["4"])

    # --- 二次执行：全部命中缓存，零函数调用 ---
    calls = []
    r2 = run_graph(store, reg, GRAPH, cache)
    assert calls == [], calls
    assert sorted(r2["cached"]) == ["1", "2", "3", "4"]
    assert r2["nodes"]["4"]["total"]["value"] == 220
    print("cache hit OK:", r2["cached"])

    # --- 修改字面量：下游失效、无关分支命中 ---
    calls = []
    g2 = {"nodes": dict(GRAPH["nodes"])}
    g2["nodes"] = {**GRAPH["nodes"], "1": {"op": "make", "inputs": {"value": 7}}}
    r3 = run_graph(store, reg, g2, cache)
    assert sorted(calls) == ["double", "make", "merge"], calls  # 3 号分支未重算
    assert r3["cached"] == ["3"], r3["cached"]
    assert r3["nodes"]["4"]["total"]["value"] == (70 * 2 + 50) * 2
    print("partial invalidation OK:", r3["cached"])

    # --- 对象过期后缓存失效，自动重算 ---
    calls = []
    store.clear()
    r4 = run_graph(store, reg, GRAPH, cache)
    assert sorted(calls) == ["double", "make", "make", "merge"], calls
    print("stale cache recompute OK")

    # --- 环检测 ---
    try:
        topo_sort({"a": {"op": "x", "inputs": {"v": ["b", "o"]}},
                   "b": {"op": "x", "inputs": {"v": ["a", "o"]}}})
        raise AssertionError("cycle not detected")
    except ValueError as e:
        print("cycle detection OK:", e)

    # --- 类型不匹配检测 ---
    store2, reg2 = ObjectStore(), build_registry()
    bad = {"nodes": {
        "1": {"op": "make", "inputs": {"value": 1}},
        "2": {"op": "merge", "inputs": {"a": ["1", "obj"], "b": ["1", "obj"], "k": 1}},
    }}
    run_graph(store2, reg2, bad, {})  # a/b 都是 LATENT，合法
    bad2 = {"nodes": {
        "1": {"op": "make", "inputs": {"value": 1}},
        "2": {"op": "double", "inputs": {"x": 123}},  # 对象端口给了字面量
    }}
    try:
        run_graph(store2, reg2, bad2, {})
        raise AssertionError("type error not detected")
    except ValueError as e:
        print("type check OK:", e)

    print("\nALL ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
