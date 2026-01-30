# 数据库设置说明

## 本地数据库挂载

PostgreSQL 数据库现在已配置为挂载到本地目录 `./pgdata`，所有数据库数据都会持久化在这个目录中。

## 使用现有备份启动

当前配置会在首次启动时自动导入 `qc_2026-01-30_12-53.dump` 备份文件。

### 启动步骤

1. **首次启动（导入备份）**
   ```bash
   # 确保 pgdata 目录为空或不存在
   rm -rf pgdata
   
   # 启动服务，会自动导入备份
   docker compose up --build
   ```

2. **后续启动（使用已有数据）**
   ```bash
   # 直接启动，不会重新导入
   docker compose up
   ```

## 重置数据库

如果需要重新导入备份或重置数据库：

```bash
# 停止服务
docker compose down

# 删除本地数据目录
rm -rf pgdata

# 重新启动（会自动导入备份）
docker compose up --build
```

## 数据目录说明

- `./pgdata/` - PostgreSQL 数据文件（本地挂载，已添加到 .gitignore）
- `./qc_2026-01-30_12-53.dump` - 数据库备份文件
- `./scripts/init-db.sh` - 自动初始化脚本

## 备份和恢复

### 创建新备份

```bash
# 导出当前数据库
docker compose exec db pg_dump -U qc -Fc qc > qc_$(date +%Y-%m-%d_%H-%M).dump
```

### 使用不同的备份文件

如果要使用其他备份文件：

1. 修改 `docker-compose.yml` 中的备份文件路径
2. 删除 `pgdata` 目录
3. 重新启动服务

## 注意事项

- ⚠️ `pgdata` 目录包含所有数据库数据，请勿手动修改
- ⚠️ 初始化脚本仅在首次启动（`pgdata` 为空）时执行
- ⚠️ 如果启动失败，检查 dump 文件是否与 PostgreSQL 16 兼容
- ✅ 数据持久化在本地，可以安全地停止和重启容器
- ✅ 可以直接备份 `pgdata` 目录来备份整个数据库
