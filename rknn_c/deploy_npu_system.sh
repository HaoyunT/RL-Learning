#!/bin/bash

echo "🚀 RK3588 NPU推理系统部署脚本"
echo "================================"
echo "当前时间: $(date)"
echo "系统信息: $(uname -a)"
echo ""

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  建议使用sudo运行部署脚本"
    echo "但系统将尝试在当前用户权限下运行..."
    sleep 2
fi

# 检查关键文件
echo "🔍 检查部署文件..."
required_files=("npu_daemon.c" "rk3588-npu-daemon.service" "compile_daemon.sh" "install_daemon.sh")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file 缺失"
        exit 1
    fi
done

# 检查模型文件
echo ""
echo "?? 检查RKNN模型文件..."
model_files=("actor_agent0.rknn" "actor_agent1.rknn" "actor_agent2.rknn" "actor_agent3.rknn")
model_count=0
for model in "${model_files[@]}"; do
    if [ -f "$model" ]; then
        echo "✅ $model (可用)"
        ((model_count++))
    else
        echo "⚠️  $model (缺失)"
    fi
done

if [ $model_count -eq 0 ]; then
    echo "❌ 未找到任何模型文件，服务将无法运行"
    exit 1
fi

echo "✅ 找到 $model_count 个模型文件"

# 编译守护进程
echo ""
echo "🔧 编译NPU守护进程..."
./compile_daemon.sh
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

# 安装服务
echo ""
echo "?? 安装系统服务..."
if [ "$EUID" -eq 0 ]; then
    ./install_daemon.sh
else
    echo "⚠️  需要root权限安装systemd服务"
    echo "请运行: sudo ./install_daemon.sh"
fi

# 环境检查
echo ""
echo "🔍 环境配置检查..."

# 检查RKNN库
RKNN_LIB_PATH="/home/khadas/khadas/rknpu2/runtime/RK3588/Linux/librknn_api/lib"
if [ -d "$RKNN_LIB_PATH" ]; then
    echo "✅ RKNN库路径: $RKNN_LIB_PATH"
    export LD_LIBRARY_PATH=$RKNN_LIB_PATH:$LD_LIBRARY_PATH
else
    echo "❌ RKNN库路径不存在"
    echo "请检查RKNPU2 SDK是否正确安装"
fi

# 检查动态库依赖
echo ""
echo "🔍 检查库依赖..."
ldd rk3588-npu-daemon 2>/dev/null | grep -E "(rknn|not found)" || echo "✅ 库依赖检查完成"

echo ""
echo "🎯 部署完成!"
echo ""
echo "📋 下一步操作:"
if [ "$EUID" -eq 0 ]; then
    echo "1. 启动服务: systemctl start rk3588-npu-daemon"
    echo "2. 查看状态: systemctl status rk3588-npu-daemon"
    echo "3. 启用自启: systemctl enable rk3588-npu-daemon"
else
    echo "1. 安装服务: sudo ./install_daemon.sh"
    echo "2. 启动服务: sudo systemctl start rk3588-npu-daemon"
    echo "3. 查看状态: sudo systemctl status rk3588-npu-daemon"
fi
echo "4. 查看日志: sudo journalctl -u rk3588-npu-daemon -f"
echo ""
echo "💡 提示: 服务将以100Hz频率持续运行NPU推理"
echo "        每1000次推理会记录一次统计信息到系统日志"]]