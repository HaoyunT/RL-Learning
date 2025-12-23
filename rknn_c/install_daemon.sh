#!/bin/bash

echo "🚀 RK3588 NPU守护进程安装脚本"
echo "================================"

# 检查权限
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用sudo运行此脚本"
    echo "用法: sudo $0"
    exit 1
fi

# 检查所需的文件
if [ ! -f "rk3588-npu-daemon" ]; then
    echo "❌ 可执行文件不存在，请先运行: ./compile_daemon.sh"
    exit 1
fi

if [ ! -f "rk3588-npu-daemon.service" ]; then
    echo "❌ 服务文件不存在"
    exit 1
fi

echo "?? 安装守护进程..."

# 复制可执行文件到系统目录
echo "复制可执行文件到 /usr/local/bin/"
cp rk3588-npu-daemon /usr/local/bin/
chmod 755 /usr/local/bin/rk3588-npu-daemon

# 复制服务文件到systemd目录
echo "复制服务文件到 /etc/systemd/system/"
cp rk3588-npu-daemon.service /etc/systemd/system/
chmod 644 /etc/systemd/system/rk3588-npu-daemon.service

# 重新加载systemd配置
echo "重新加载systemd配置..."
systemctl daemon-reload

# 创建必要的目录
echo "创建运行时目录..."
mkdir -p /var/run
touch /var/run/rk3588-npu-daemon.pid
chmod 644 /var/run/rk3588-npu-daemon.pid

echo "✅ 安装完成！"

echo ""
echo "📋 使用说明:"
echo "启动服务:     sudo systemctl start rk3588-npu-daemon"
echo "停止服务:     sudo systemctl stop rk3588-npu-daemon"
echo "重启服务:     sudo systemctl restart rk3588-npu-daemon"
echo "查看状态:     sudo systemctl status rk3588-npu-daemon"
echo "开机自启:     sudo systemctl enable rk3588-npu-daemon"
echo "禁用自启:     sudo systemctl disable rk3588-npu-daemon"
echo ""
echo "📊 日志查看:"
echo "实时日志:     sudo journalctl -u rk3588-npu-daemon -f"
echo "最近日志:     sudo journalctl -u rk3588-npu-daemon --since \"1 hour ago\""
echo "全部日志:     sudo journalctl -u rk3588-npu-daemon"

echo ""
echo "🔧 测试服务:"
echo "sudo systemctl start rk3588-npu-daemon"
echo "sudo systemctl status rk3588-npu-daemon"
echo "如果状态正常，可以设置开机自启"]]