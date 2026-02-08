# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 FastAPI 的客服质检与培训系统（MVP），支持：
- 多角色登录（客服/客服主管/管理员）
- 导入 AI 对话分析 JSON 生成报表批次
- 对话详情查看和训练任务创建
- 客服改写练习提交和主管评分反馈
- 标签管理与合并功能（新增：2026-01-30）
- **自动质检与未绑定账号通知（新增：2026-02-02）**
 - 导入后自动检测未绑定客服账号并飞书通知
 - 每小时自动检查符合条件的对话并触发AI质检
 - 详见：`docs/AUTO_QC_QUICKSTART.md`
- **客服分析报表（新增：2026-02-02）**
 - 多维度交叉分析：客服 → 日期 → 场景 → 满意度
 - 支持按天/周/月分组统计
 - 详见：`docs/AGENT_ANALYSIS_REPORT.md`

## 快速启动

```bash
# 一键启动（推荐）
docker compose up --build

# 访问地址
http://localhost:8000

# 默认账号（登录使用「用户名」，不是邮箱）
# docker-compose 默认先创建 bootstrap 管理员，再按 SEED_DEMO 决定是否种子演示用户
管理员：用户名 Sean / 密码 0357zaqxswcde（来自 DEFAULT_ADMIN_USERNAME、DEFAULT_ADMIN_PASSWORD）
# 仅当 SEED_DEMO=1 且数据库为空时才会种子以下演示账号（当前流程通常先有 Sean，故常仅管理员）
# 主管：supervisor / supervisor123；客服：agent1 / agent123、agent2 / agent123
```

## 开发命令

### 本地开发
```bash
# 安装依赖
cd app
pip install -r requirements.txt

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 运行后台worker
python worker.py

# 运行定时任务
python cron_runner.py
```

### 测试/运行规则（重要）
- **所有测试与运行必须在 Docker 容器内执行**（例如：`docker compose exec app ...`）。
- 避免在宿主机直接运行 `python`/`pytest`/`uvicorn`。

### 数据库操作
```bash
# 首次启动（自动导入备份）
# 确保 pgdata 目录为空
rm -rf pgdata
docker compose up --build

# 重建数据库（谨慎使用）
docker compose down
rm -rf pgdata
docker compose up --build

# 迁移数据库
python migrate.py

# 重置并重新导入数据
bash scripts/reset_all_chats_and_reimport.sh

# 创建数据库备份
docker compose exec db pg_dump -U qc -Fc qc > qc_$(date +%Y-%m-%d_%H-%M).dump

# 详细说明
# 参考 DATABASE_SETUP.md 获取完整的数据库设置说明
```

### 导入数据
```bash
# 生成测试JSON数据
python scripts/generate_fake_json.py

# 使用示例数据路径
data/sample_batch.json
```

## 架构概览

### 核心组件

**Backend (FastAPI)**
- `app/main.py` - 主应用入口，4K+ 行代码，包含所有路由和业务逻辑
- `app/models.py` - SQLModel 数据模型定义
- `app/auth.py` - 认证和权限管理
- `app/db.py` - 数据库连接和会话管理

**数据导入**
- `app/importers/json_import.py` - JSON 文件导入器
- `app/importers/leyan_import.py` - 乐言平台导入器
- `app/bucket_import.py` - S3 兼容存储桶导入（阿里 OSS；支持同桶混合淘宝/抖音自动识别）

**AI 集成**
- `app/ai_client.py` - OpenAI API 客户端（通过 relay 转发）
- `app/prompts.py` - AI 分析提示词
- `app/analysis_engine.py` - AI 分析引擎

**后台任务**
- `app/worker.py` - 后台任务处理器
- `app/cron_runner.py` - 定时任务调度器（含自动质检检查）
- `app/daily_summary.py` - AI 日报生成
- `app/qc_checker.py` - 质检检查器（新增：2026-02-02）

**数据模型 (models.py)**
```
User (用户)
├── Role: admin/supervisor/agent
└── 关联：Conversation, TrainingTask

UserThread (用户线程)
├── 跨天对话分组
└── 关联：Conversation

Conversation (对话)
├── 关联：Message, AnalysisBatch, UserThread
└── 关联：AgentBinding

Message (消息)
├── 对话消息内容
└── 关联：Conversation

AnalysisBatch (分析批次)
├── 批次元信息
└── 关联：ConversationAnalysis

ConversationAnalysis (对话分析)
├── AI 分析结果
└── 关联：Conversation, AnalysisBatch

TrainingTask (训练任务)
├── 任务创建和分配
└── 关联：TrainingAttempt

TrainingAttempt (训练尝试)
├── 客服提交的改写练习
└── 关联：TrainingTask
```

### 服务架构 (docker-compose.yml)

```
┌─────────────────┐
│   PostgreSQL    │ ← 数据库 (端口 5432)
│    (db)         │
└─────────┬───────┘
          │
          ▼
    ┌──────────┐
    │   App    │ ← FastAPI 主服务 (端口 8000)
    │ (app)    │
    └────┬─────┘
         │
         ▼ (异步队列)
    ┌──────────┐
    │  Worker  │ ← 后台任务处理器
    │(worker)  │
    └──────────┘
         │
         ▼ (定时任务)
    ┌──────────┐
    │   Cron   │ ← 定时导入任务
    │  (cron)  │
    └──────────┘
```

## 关键文件位置

### 路由和业务逻辑
- `app/main.py` - 所有 API 路由和页面处理
- `app/templates/` - Jinja2 HTML 模板
- `app/static/app.css` - 静态样式

### 配置
- `.env.example` - 环境变量模板
- `docker-compose.yml` - 服务配置
- `app/app_config.py` - 应用配置

### 工具脚本
- `scripts/reset_all_chats_and_reimport.sh` - 重置和重新导入数据
- `scripts/generate_fake_json.py` - 生成测试数据
- `scripts/shift_taobao_timestamps_plus8.ps1` - 时间戳处理工具

### 导入格式
JSON 文件结构 (`data/sample_batch.json`):
```json
{
  "batch_meta": {
    "platform": "taobao",
    "date": "2026-01-19",
    "source": "leyan"
  },
  "conversations": [
    {
      "conversation_id": "xxx",
      "messages": [...],
      "analysis": {
        "score": 85,
        "issues": [...],
        "summary": "..."
      }
    }
  ]
}
```

## 环境变量

### 必需配置
```bash
# 数据库
POSTGRES_USER=qc
POSTGRES_PASSWORD=qc
POSTGRES_DB=qc

# 应用
APP_SECRET_KEY=change-me-in-prod
SEED_DEMO=1
OPEN_REGISTRATION=0

# AI 配置（通过 relay 转发）
OPENAI_BASE_URL=http://www.marllen.com:4000/v1
OPENAI_MODEL=chatgpt-5.1
OPENAI_API_KEY=your_key
OPENAI_TIMEOUT=1800

# 阿里 OSS (可选)
ENABLE_TAOBAO_BUCKET_IMPORT=1
TAOBAO_BUCKET_NAME=conversations-marllen-taobao
TAOBAO_S3_ENDPOINT=https://oss-cn-shanghai.aliyuncs.com
TAOBAO_S3_ACCESS_KEY=your_key
TAOBAO_S3_SECRET_KEY=your_secret
TAOBAO_BUCKET_PREFIX=leyan

# 飞书通知 (可选)
FEISHU_WEBHOOK_URL=your_webhook_url
```

## 常见开发任务

### 添加新页面
1. 在 `app/main.py` 中添加路由
2. 在 `app/templates/` 创建 HTML 模板
3. 添加对应的业务逻辑和权限检查

### 修改数据模型
1. 更新 `app/models.py` 中的 SQLModel 定义
2. 运行 `python app/migrate.py` 迁移数据库
3. 更新相关导入器 (`app/importers/`)

### 添加新的导入器
1. 在 `app/importers/` 创建新的导入模块
2. 参考 `json_import.py` 和 `leyan_import.py` 的实现
3. 在 `bucket_import.py` 中注册

### AI 分析定制
1. 修改 `app/prompts.py` 中的提示词
2. 调整 `app/analysis_engine.py` 中的分析逻辑
3. 配置 `app/ai_client.py` 中的 API 调用

### 后台任务
1. 创建任务处理器 (`app/worker.py` 的 `handle_analysis_job` 参考)
2. 在主应用中入队任务
3. Worker 自动处理任务队列

### 标签管理
1. **标签合并**：`POST /settings/tags/tag/merge`
 - 将源标签的所有对话标记迁移到目标标签
 - 自动处理去重（同一对话同时有两个标签时保留目标标签）
 - 迁移 AI 命中记录（ConversationTagHit）和手动标记（ManualTagBinding）
 - 完成后删除源标签
 - 详见：`TAG_MERGE_USAGE.md` 和 `docs/TAG_MERGE_FEATURE.md`
2. **停用标签**：`POST /settings/tags/tag/delete` - 软删除
3. **删除标签**：`POST /settings/tags/tag/remove` - 硬删除（包括所有命中记录）
4. **批量导入**：`POST /settings/tags/import` - 从 Excel 导入标签配置
5. **已驳回的标签建议**：TagSuggestion 中 status=rejected 的记录会动态加入AI质检提示词，指导AI勿再建议类似标签（无需单独管理界面）

### 自动质检（新增：2026-02-02）
1. **未绑定客服账号检测**
 - 导入后自动检测未绑定的外部客服账号
 - 通过飞书 webhook 发送通知（包含客服昵称）
 - 去重机制，仅通知新出现的未绑定账号
2. **自动质检触发**
 - 每小时检查一次符合条件的对话（>5消息 且 所有客服已绑定）
 - 导入成功后立即触发检查（无需等待整点）
 - 自动创建 AIAnalysisJob 任务并入队
3. **使用说明**
 - 快速开始：`docs/AUTO_QC_QUICKSTART.md`
 - 详细文档：`docs/AUTO_QC_FEATURE.md`
 - 更新日志：`CHANGELOG_AUTO_QC.md`
 - 测试脚本：`scripts/test_auto_qc.py`

## 调试指南

### 日志查看
```bash
# 查看应用日志
docker compose logs -f app

# 查看worker日志
docker compose logs -f worker

# 查看cron日志
docker compose logs -f cron
```

### 数据库调试
```bash
# 连接数据库
docker compose exec db psql -U qc -d qc

# 查看表结构
\d+ table_name

# 查看数据
SELECT * FROM table_name LIMIT 10;
```

### 常见问题

**1. 500 错误**
- 检查数据库连接
- 查看应用日志：`docker compose logs app`
- 验证环境变量配置

**2. 导入失败**
- 验证 JSON 格式是否符合 `sample_batch.json`
- 检查文件路径权限
- 查看 worker 日志：`docker compose logs worker`

**3. AI 分析失败**
- 验证 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`
- 检查网络连接和代理配置
- 查看超时设置：`OPENAI_TIMEOUT`

## 依赖关系

### Python 核心依赖
- `fastapi==0.115.4` - Web 框架
- `uvicorn[standard]==0.32.0` - ASGI 服务器
- `sqlmodel==0.0.22` - ORM（SQLAlchemy + Pydantic）
- `psycopg[binary]==3.2.3` - PostgreSQL 驱动
- `jinja2==3.1.4` - 模板引擎
- `openpyxl==3.1.5` - Excel 文件处理
- `boto3==1.34.162` - AWS S3 客户端（用于 OSS/S3 兼容桶）

### 开发工具
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 16

## 重要提醒

1. **数据库迁移**：修改模型后务必运行 `python app/migrate.py`
2. **环境变量**：生产环境必须修改 `APP_SECRET_KEY` 和数据库密码
3. **文件权限**：Docker 容器内 `./data` 挂载到 `/data`，确保有写入权限
4. **AI 集成**：通过 relay 转发 OpenAI API，配置 `OPENAI_BASE_URL`
5. **数据备份**：`./data` 目录包含导入的备份文件，定期备份
6. **角色权限**：User 模型支持 `admin/supervisor/agent` 三种角色
7. **软删除**：用户删除使用 `deleted_at` 字段软删除，保留历史数据
8. **后台任务**：AI 分析和日报生成都在后台异步执行，不要同步等待
