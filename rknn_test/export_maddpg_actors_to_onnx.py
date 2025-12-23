import os
import torch
import torch.nn as nn

# ===============================
# 你模型的真实结构（必须和训练一致）
# ===============================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.pi = nn.Linear(128, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.pi(x))


# ===============================
# 主导出逻辑
# ===============================
def export_all_actors_to_onnx():
    # 当前脚本所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] Base dir: {BASE_DIR}")

    # ===== 根据你现在已确认的真实维度 =====
    # OBS_DIM = 26   # ← 你前面已经跑通的维度（非常关键）
    ACT_DIM = 2

    # 4 个 agent
    NUM_AGENTS = 4
    
    # 不同 agent 可能有不同的 obs_dim
    # agent 0-2: 26, agent 3: 23
    OBS_DIMS = [26, 26, 26, 23]

    for i in range(NUM_AGENTS):
        actor_path = os.path.join(BASE_DIR, f"agent_{i}_actor")
        onnx_path = os.path.join(BASE_DIR, f"actor_agent{i}.onnx")
        
        # 获取当前 agent 的 obs_dim
        current_obs_dim = OBS_DIMS[i]

        print(f"\n[INFO] Processing agent {i}")
        print(f"       actor checkpoint: {actor_path}")
        print(f"       obs_dim: {current_obs_dim}")

        if not os.path.exists(actor_path):
            print(f"[ERROR] File not found: {actor_path}")
            continue

        # 1. 建 actor 网络
        actor = Actor(current_obs_dim, ACT_DIM)

        # 2. 加载权重
        state_dict = torch.load(actor_path, map_location="cpu")
        actor.load_state_dict(state_dict)
        actor.eval()

        print("[OK] Actor loaded")

        # 3. dummy input（batch=1）
        dummy_input = torch.randn(1, current_obs_dim)

        # 4. 导出 ONNX
        torch.onnx.export(
            actor,
            dummy_input,
            onnx_path,
            input_names=["obs"],
            output_names=["action"],
            opset_version=11,
            do_constant_folding=True,
            dynamic_axes=None
        )

        print(f"[OK] Exported: {onnx_path}")

    print("\n🎉 All actors exported to ONNX successfully!")


if __name__ == "__main__":
    export_all_actors_to_onnx()
