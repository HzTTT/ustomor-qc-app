# 数据库配置更新说明

## 更新内容

### 1. PostgreSQL 数据本地挂载
- **之前**: 使用 Docker volume (`db_data`)
- **现在**: 挂载到本地目录 `./pgdata`
- **优势**: 
  - 数据直接存储在项目目录
  - 便于备份和迁移
  - 可以直接访问数据文件

### 2. 自动导入数据库备份
- 在首次启动时自动导入 `qc_2026-01-30_12-53.dump`
- 无需手动执行导入命令
- 只在数据库为空时执行（避免重复导入）

### 3. 新增文件

#### `scripts/init-db.sh`
数据库初始化脚本，在容器首次启动时自动执行：
- 检测备份文件
- 使用 `pg_restore` 导入数据
- 提供详细的日志输出

#### `DATABASE_SETUP.md`
完整的数据库设置文档，包含：
- 启动步骤
- 重置数据库方法
- 备份和恢复指南
- 注意事项

#### `start.sh`
便捷启动脚本：
- 自动检测是否有现有数据
- 提供重置数据库选项
- 友好的用户提示

#### `.gitignore`
添加了合理的忽略规则：
- `pgdata/` - 数据库数据目录
- Python 缓存文件
- 环境变量文件
- 临时文件

### 4. 更新文件

#### `docker-compose.yml`
- 修改数据库卷挂载：`./pgdata:/var/lib/postgresql/data`
- 添加备份文件挂载：`./qc_2026-01-30_12-53.dump:/docker-entrypoint-initdb.d/restore.dump`
- 添加初始化脚本挂载：`./scripts/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh`
- 移除 Docker volume 定义

#### `CLAUDE.md`
- 更新数据库操作说明
- 添加备份创建命令
- 添加数据库设置文档引用

## 使用方法

### 快速启动（推荐）

```bash
# 使用启动脚本
./start.sh
```

### 手动启动

```bash
# 首次启动（自动导入备份）
docker compose up --build

# 后续启动
docker compose up
```

### 重置数据库

```bash
# 停止服务
docker compose down

# 删除数据目录
rm -rf pgdata

# 重新启动（自动重新导入）
docker compose up --build
```

## 迁移现有安装

如果你之前已经在使用这个项目：

1. **停止服务**
   ```bash
   docker compose down
   ```

2. **（可选）导出现有数据**
   ```bash
   docker compose up -d db
   docker compose exec db pg_dump -U qc -Fc qc > my_backup.dump
   docker compose down
   ```

3. **删除旧的 volume**
   ```bash
   docker volume rm customor-qc-app_20260130_124735_db_data
   ```

4. **启动新配置**
   ```bash
   docker compose up --build
   ```

## 注意事项

- ⚠️ 首次启动会自动导入 `qc_2026-01-30_12-53.dump`
- ⚠️ 如果 `pgdata` 目录已存在，不会重新导入
- ⚠️ 备份文件必须与 PostgreSQL 16 兼容
- ✅ 可以安全地使用 `docker compose down` 而不丢失数据
- ✅ `pgdata` 目录已添加到 `.gitignore`
- ✅ 数据库端口仍然是 `5433` (主机) -> `5432` (容器)

## 故障排除

### 问题：启动时卡在 "等待 PostgreSQL 启动"
**解决方案**: 检查 dump 文件是否损坏或不兼容

### 问题：pg_restore 报错
**解决方案**: 某些警告是正常的（如对象已存在），只要服务能启动就可以

### 问题：想使用其他备份文件
**解决方案**: 
1. 修改 `docker-compose.yml` 中的备份文件路径
2. 删除 `pgdata` 目录
3. 重新启动

## 更多信息

详细的数据库设置说明请参考：`DATABASE_SETUP.md`
