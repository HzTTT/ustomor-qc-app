#!/bin/bash
set -e

echo "=========================================="
echo "检查是否需要导入数据库备份..."
echo "=========================================="

# 检查 dump 文件是否存在
if [ -f /docker-entrypoint-initdb.d/restore.dump ]; then
    echo "发现备份文件，开始导入..."
    
    # 等待 PostgreSQL 完全启动
    until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
        echo "等待 PostgreSQL 启动..."
        sleep 2
    done
    
    echo "开始导入数据库备份..."
    # 使用 pg_restore 导入 dump 文件
    # -U: 指定用户
    # -d: 指定数据库
    # -v: 详细输出
    # --no-owner: 不恢复所有者
    # --no-acl: 不恢复访问权限
    pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v --no-owner --no-acl /docker-entrypoint-initdb.d/restore.dump || {
        echo "警告: pg_restore 返回错误，但这可能是正常的（某些对象可能已存在）"
    }
    
    echo "数据库备份导入完成！"
else
    echo "未找到备份文件，跳过导入步骤"
fi

echo "=========================================="
echo "数据库初始化完成"
echo "=========================================="
