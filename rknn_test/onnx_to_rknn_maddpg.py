from rknn.api import RKNN

AGENT_IDS = [0,1,2,3]   # 只转 3 个 UAV 的 actor，如需包含 target，改成 [0,1,2,3]

def convert_one_agent(agent_id):
    rknn = RKNN(verbose=True)

    onnx_path = f'actor_agent{agent_id}.onnx'
    rknn_path = f'actor_agent{agent_id}.rknn'

    print(f'\n=== Agent {agent_id}: ONNX -> RKNN ===')

    # 1. 配置：目标平台 rk3588，RL 输入是向量，不需要 mean/std
    print('--> Config')
    rknn.config(
        target_platform='rk3588',
        mean_values=None,
        std_values=None
    )
    print('done')

    # 2. 加载 ONNX
    print('--> Load ONNX:', onnx_path)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load onnx failed!')
        exit(ret)
    print('done')

    # 3. 构建 RKNN 模型（先不量化，保证和 PyTorch 一致性）
    print('--> Build RKNN (no quant)')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build rknn failed!')
        exit(ret)
    print('done')

    # 4. 导出 .rknn
    print('--> Export RKNN:', rknn_path)
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()
    print(f'=== Agent {agent_id} 完成，生成 {rknn_path} ===')

def main():
    for aid in AGENT_IDS:
        convert_one_agent(aid)

if __name__ == '__main__':
    main()
