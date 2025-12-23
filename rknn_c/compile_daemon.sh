#!/bin/bash

echo "🔧 编译RK3588 NPU守护进程"

# 检查环境
RKNN_API_PATH="/home/khadas/khadas/rknpu2/runtime/RK3588/Linux/librknn_api"

if [ ! -d "$RKNN_API_PATH" ]; then
    echo "❌ RKNN API路径不存在: $RKNN_API_PATH"
    echo "请先安装RKNPU2 SDK"
    exit 1
fi

echo "✅ 使用RKNN API路径: $RKNN_API_PATH"

# 编译守护进程
echo "📦 编译守护进程..."
gcc npu_daemon.c -o rk3588-npu-daemon \
    -I"$RKNN_API_PATH/include" \
    -L"$RKNN_API_PATH/lib" \
    -lrknnrt -lm

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
else
    echo "❌ 编译失败"
    exit 1
fi

echo ""
echo "安装说明:"
echo "1. 安装到系统: sudo ./install_daemon.sh"
echo "2. 启动服务: sudo systemctl start rk3588-npu-daemon"
echo "3. 启用开机自启: sudo systemctl enable rk3588-npu-daemon"
echo "4. 查看状态: sudo systemctl status rk3588-npu-daemon"
echo "5. 查看日志: sudo journalctl -u rk3588-npu-daemon -f"]]