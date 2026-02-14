#!/bin/bash

echo "=========================================="
echo "客服质检系统 - 启动脚本"
echo "=========================================="
echo ""

# 构建本地 Tailwind CSS，避免依赖外网 CDN（在国内环境经常不稳定/被拦截）。
if command -v npm >/dev/null 2>&1; then
    echo "构建前端样式 (Tailwind)..."
    npm run build:tailwind >/dev/null 2>&1 || echo "⚠️  Tailwind 构建失败（将继续启动，但界面可能缺少样式）"
else
    echo "⚠️  未检测到 npm，跳过 Tailwind 构建（界面可能缺少样式）"
fi

if [ ! -f "app/static/tailwind.css" ]; then
    echo "❌ 未找到 app/static/tailwind.css。为避免出现“只有文本没 UI”，本次启动中止。"
    echo "   解决办法：安装 Node/npm 后执行：npm install && npm run build:tailwind"
    exit 1
fi

# 检查是否存在 pgdata 目录
if [ -d "pgdata" ]; then
    echo "✓ 检测到已有数据库数据目录 (pgdata/)"
    echo "  将使用现有数据库数据启动"
    echo ""
    read -p "是否要重置数据库并重新导入备份？(y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在删除现有数据库..."
        docker compose down
        rm -rf pgdata
        echo "✓ 数据库已重置"
        echo ""
    fi
else
    echo "✓ 未检测到数据库数据目录"
    echo "  首次启动将自动导入备份: qc_2026-01-30_12-53.dump"
    echo ""
fi

echo "启动 Docker 容器 (worker=4)..."
echo "=========================================="
docker compose up --build --scale worker=4 -d

# 如果启动失败，显示帮助信息
if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "启动失败！请检查："
    echo "1. Docker 是否正在运行"
    echo "2. 端口 8000 和 5433 是否被占用"
    echo "3. .env 文件配置是否正确"
    echo ""
    echo "查看详细日志："
    echo "  docker compose logs -f"
    echo ""
    echo "查看数据库设置说明："
    echo "  cat DATABASE_SETUP.md"
    echo "=========================================="
fi
