# 自动质检功能说明

本文档说明新增的自动质检相关功能。

## 功能概述

### 1. 未绑定客服账号检测与通知

**功能描述：**
- 当导入聊天记录（batch）后，系统会自动检测是否存在未绑定的外部客服账号
- 如果发现未绑定的客服账号，会通过飞书 webhook 发送通知
- 通知消息包含客服账号的昵称（`assistant_nick`）作为提示信息

**触发时机：**
1. 定时任务自动从 OSS/TOS 导入时
2. 管理员手动触发导入时（通过【设置 > 对象存储设置】页面）

**通知格式：**
```
⚠️ 发现未绑定客服账号

平台：taobao
日期：2026-01-31

未绑定账号（新增）：
• account_001 (昵称: 张三)
• account_002 (昵称: 李四)

请到【设置 > 客服账号绑定】页面进行配置。
```

**实现细节：**
- 系统会记录已通知过的未绑定账号，避免重复通知
- 只有新出现的未绑定账号才会触发通知
- 昵称信息从聊天记录中的 `assistant_nick` 字段获取，可作为创建绑定时的默认姓名

### 2. 自动质检检查与触发

**功能描述：**
- 系统每小时自动检查一次，查找符合自动质检条件的对话
- 符合条件时，自动创建AI质检任务并入队

**质检条件：**
1. 对话尚未进行过质检（没有 `ConversationAnalysis` 记录）
2. 对话消息数 > 5 条
3. 对话中涉及的所有客服账号都已成功绑定到站内用户

**触发时机：**
1. 每小时自动检查一次（由 cron 定时任务执行）
2. 每次导入成功后立即检查一次（不需要等待下一个整点）

**工作流程：**
```
导入聊天记录
    ↓
检查是否有未绑定客服 → 有 → 发送飞书通知
    ↓
    否
    ↓
检查对话是否符合质检条件
    ↓
符合条件 → 创建 AIAnalysisJob 任务
    ↓
后台 worker 自动处理质检任务
    ↓
质检完成，结果存入 ConversationAnalysis
```

**通知格式（当发现可质检对话时）：**
```
✅ 自动质检已启动

发现 15 个符合质检条件的对话
已为 15 个对话创建AI质检任务

条件：消息数>5 且 所有客服已绑定
```

## 配置说明

### 飞书 Webhook 配置

在【设置 > 对象存储设置】页面配置飞书 webhook URL：

```
https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-token
```

或在 `.env` 文件中配置：

```bash
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook-token
```

### 自动质检开关

自动质检功能依赖以下配置项（在【设置 > 通用设置】页面）：

- **自动分析开关**：`auto_analysis_enabled` - 必须开启
- **自动入队开关**：`enable_enqueue_analysis` - 必须开启

## 技术实现

### 新增文件

1. **`app/qc_checker.py`** - 质检检查器模块
   - `get_unbound_agent_info()` - 获取未绑定客服账号详细信息
   - `get_conversations_eligible_for_qc()` - 获取符合自动质检条件的对话列表

### 修改文件

1. **`app/importers/leyan_import.py`**
   - 导入时收集未绑定客服账号的昵称信息
   - 返回结果中新增 `unbound_agent_nicks` 字段

2. **`app/bucket_import.py`**
   - 聚合多个导入文件的未绑定账号信息
   - 返回结果中包含 `unbound_agent_nicks` 字段

3. **`app/cron_runner.py`**
   - 改进未绑定客服账号通知（包含昵称）
   - 新增每小时自动质检检查逻辑
   - 导入成功后立即触发质检检查

4. **`app/main.py`**
   - 手动导入时也会检测并通知未绑定客服账号

## 监控与日志

### 查看定时任务日志

1. 通过【设置 > 对象存储设置】页面查看导入日志
2. 日志中会记录自动质检检查的结果

### 查看质检任务队列

```sql
-- 查看待处理的质检任务
SELECT * FROM aianalysisjob WHERE status = 'pending' ORDER BY created_at;

-- 查看正在处理的质检任务
SELECT * FROM aianalysisjob WHERE status = 'running' ORDER BY started_at DESC;

-- 查看最近完成的质检任务
SELECT * FROM aianalysisjob WHERE status = 'done' ORDER BY finished_at DESC LIMIT 10;
```

### 查看符合质检条件的对话

```sql
-- 查看未质检且消息数>5的对话
SELECT 
    c.id,
    c.external_id,
    c.platform,
    c.agent_account,
    COUNT(m.id) as message_count
FROM conversation c
LEFT JOIN message m ON m.conversation_id = c.id
LEFT JOIN conversationanalysis ca ON ca.conversation_id = c.id
WHERE ca.id IS NULL
GROUP BY c.id
HAVING COUNT(m.id) > 5
ORDER BY c.id DESC
LIMIT 20;
```

## 常见问题

### Q1: 为什么有些对话符合条件但没有自动质检？

**A:** 检查以下几点：
1. 确认所有涉及的客服账号都已绑定（查看【设置 > 客服账号绑定】）
2. 确认自动分析开关已开启（【设置 > 通用设置】）
3. 检查后台 worker 是否正常运行（`docker compose logs worker`）

### Q2: 如何手动触发质检？

**A:** 有两种方式：
1. 在对话详情页点击【触发AI质检】按钮
2. 使用 API：`POST /conversations/{conversation_id}/analyze`

### Q3: 质检任务会自动重试吗？

**A:** 
- 如果任务失败，不会自动重试
- 如果 worker 崩溃导致任务卡在 `running` 状态，30分钟后会自动标记为 `error`
- 可以通过重启 worker 来恢复处理

### Q4: 如何调整自动质检的消息数阈值？

**A:** 
- 当前阈值硬编码为 5 条消息
- 如需调整，修改 `cron_runner.py` 中的 `min_messages` 参数
- 建议值：5-10 条（太少可能质量不佳，太多会遗漏短对话）

## 测试方法

### 测试未绑定客服账号通知

1. 导入包含新客服账号的聊天记录（未绑定）
2. 检查飞书群是否收到通知
3. 通知中应包含客服账号和昵称

### 测试自动质检

1. 导入聊天记录（>5条消息）
2. 绑定所有涉及的客服账号
3. 等待最多1小时（或手动触发导入）
4. 检查是否收到自动质检启动的通知
5. 查看质检任务队列：`SELECT * FROM aianalysisjob WHERE status='pending';`
6. 等待 worker 处理完成
7. 查看质检结果：对话详情页会显示AI分析结果

## 性能考虑

- 每小时检查限制最多 50 个对话（可调整 `limit` 参数）
- 每次批量创建任务时会检查是否已存在，避免重复入队
- 未绑定账号记录会持久化，避免重复通知
- 所有检查操作都在后台定时任务中执行，不影响用户界面响应

## 后续优化建议

1. 支持自定义消息数阈值（通过 AppConfig）
2. 支持按平台/时间段过滤符合条件的对话
3. 添加质检任务优先级机制
4. 支持批量重试失败的质检任务
5. 提供质检进度仪表板
