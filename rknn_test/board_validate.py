import time
import numpy as np
from rknnlite.api import RKNNLite

# =========================
# 4 个智能体配置
# =========================
AGENTS = {
    0: {
        "rknn": "./actor_agent0.rknn",
        "obs": "./obs_actor_0.npy",
        "torch_out": "./torch_out_actor_0.npy",
    },
    1: {
        "rknn": "./actor_agent1.rknn",
        "obs": "./obs_actor_1.npy",
        "torch_out": "./torch_out_actor_1.npy",
    },
    2: {
        "rknn": "./actor_agent2.rknn",
        "obs": "./obs_actor_2.npy",
        "torch_out": "./torch_out_actor_2.npy",
    },
    3: {
        "rknn": "./actor_agent3.rknn",
        "obs": "./obs_actor_3.npy",
        "torch_out": "./torch_out_actor_3.npy",
    },
}

LOOP = 100   # 性能统计循环次数

def load_rknn(rknn_path: str) -> RKNNLite:
    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    assert ret == 0, f"load_rknn failed: {rknn_path}"
    ret = rknn.init_runtime(target="rk3588")
    assert ret == 0, f"init_runtime failed: {rknn_path}"
    return rknn

def main():
    print("========== 4-Agents Numeric Validation + Performance ==========")

    # 1️⃣ 初始化所有 RKNN
    agents = {}
    for i, cfg in AGENTS.items():
        obs = np.load(cfg["obs"]).astype(np.float32)
        torch_out = np.load(cfg["torch_out"]).astype(np.float32)

        rknn = load_rknn(cfg["rknn"])

        # warmup
        _ = rknn.inference(inputs=[obs])

        agents[i] = {
            "rknn": rknn,
            "obs": obs,
            "torch_out": torch_out,
        }

        print(f"[Init] agent_{i}:")
        print(f"       obs shape = {obs.shape}")
        print(f"       torch_out shape = {torch_out.shape}")

    # 2️⃣ 数值验证（逐 agent）
    print("\n========== Numeric Consistency ==========")
    for i, info in agents.items():
        rknn = info["rknn"]
        obs = info["obs"]
        torch_out = info["torch_out"]

        out = rknn.inference(inputs=[obs])
        rknn_out = np.array(out[0], dtype=np.float32)
        np.save(f"rknn_out_actor_{i}.npy", rknn_out)

        abs_err = np.abs(rknn_out - torch_out)
        rel_err = abs_err / (np.abs(torch_out) + 1e-6)

        print(f"\n[agent_{i}]")
        print("  rknn_out =", rknn_out)
        print("  torch_out =", torch_out)
        print("  Max Abs Error =", float(abs_err.max()))
        print("  Max Rel Error =", float(rel_err.max()))

    # 3️⃣ 性能统计（4 actor 串行）
    print("\n========== Performance (Serial 4 Actors) ==========")

    t0 = time.time()
    for _ in range(LOOP):
        for info in agents.values():
            _ = info["rknn"].inference(inputs=[info["obs"]])
    t1 = time.time()

    total = t1 - t0
    avg_total_ms = total / LOOP * 1000
    avg_one_ms = avg_total_ms / len(agents)
    fps = LOOP / total

    print(f"LOOP = {LOOP}")
    print(f"Avg total latency (4 agents) = {avg_total_ms:.3f} ms")
    print(f"Avg per-agent latency = {avg_one_ms:.3f} ms")
    print(f"Control FPS ≈ {fps:.2f}")

    # 4️⃣ release
    for info in agents.values():
        info["rknn"].release()

    print("\n========== DONE ==========")

if __name__ == "__main__":
    main()
