#!/bin/bash

echo "🔧 编译RK3588 RKNN模型验证工具"

# 检查RKNN API路径
RKNN_API_PATH="/home/khadas/khadas/rknpu2/runtime/RK3588/Linux/librknn_api/include"

if [ ! -d "$RKNN_API_PATH" ]; then
    echo "❌ RKNN API路径不存在: $RKNN_API_PATH"
    echo "请检查RKNPU2是否已正确安装"
    exit 1
fi

echo "✅ 使用RKNN API路径: $RKNN_API_PATH"

# 编译验证程序
echo "📦 编译 rknn_validate.c..."
gcc rknn_validate.c -o validate_rknn \
    -I"$RKNN_API_PATH" \
    -lrknnrt -lm

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "使用说明："
    echo "1. 直接运行验证: ./validate_rknn"
    echo "2. 验证单个模型: ./validate_rknn model_path.rknn"
    echo "3. 批量验证所有Agent模型"
    echo ""
    echo "验证功能："
    echo "- 模型文件检查"
    echo "- RKNN SDK兼容性测试"
    echo "- 推理性能基准测试 (100次迭代)"
    echo "- 实时性能评估 (是否满足100Hz)"
    echo "- 批量模型验证报告"
else
    echo "❌ 编译失败"
    exit 1
fi