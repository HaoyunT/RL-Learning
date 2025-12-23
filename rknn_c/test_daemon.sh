#!/bin/bash

echo "🧪 RK3588 NPU守护进程功能测试"
echo "================================"

# 检查守护进程可执行文件
if [ ! -f "rk3588-npu-daemon" ]; then
    echo "❌ 守护进程文件不存在，请先运行: ./compile_daemon.sh"
    exit 1
fi

echo "✅ 守护进程文件检查: rk3588-npu-daemon"

# 检查模型文件
if [ ! -f "actor_agent0.rknn" ]; then
    echo "❌ 模型文件不存在: actor_agent0.rknn"
    exit 1
fi
echo "✅ 模型文件检查: actor_agent0.rknn"

# 设置库路径
export LD_LIBRARY_PATH=/lib:$LD_LIBRARY_PATH

echo ""
echo "🔍 动态库检查:"
ldd rk3588-npu-daemon | grep rknn

echo ""
echo "🏃‍♂️ 启动守护进程测试（非守护模式）..."
echo "服务将在前台运行，按 Ctrl+C 停止"
echo "日志将直接输出到终端"
echo "================================"
echo ""

# 在前台运行守护进程（非守护模式，便于测试）
./rk3588-npu-daemon
