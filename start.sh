#!/bin/bash

echo "=========================================="
echo "客服质检系统 - 启动脚本"
echo "=========================================="
echo ""

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

echo "启动 Docker 容器..."
echo "=========================================="
docker compose up --build

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
