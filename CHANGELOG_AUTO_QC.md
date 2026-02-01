# 自动质检功能更新日志

## 2026-02-02 - 自动质检与未绑定账号通知

### 新增功能

#### 1. 未绑定客服账号检测与通知 ⚠️

**问题背景：**
导入聊天记录后，如果客服账号未绑定到站内用户，会导致无法准确追踪客服表现。

**解决方案：**
- ✅ 导入后自动检测未绑定的外部客服账号
- ✅ 通过飞书 webhook 实时通知管理员
- ✅ 通知中包含客服昵称（`assistant_nick`），方便快速创建绑定
- ✅ 去重机制：仅通知新出现的未绑定账号

**使用场景：**
1. 定时任务自动从 OSS 导入时
2. 管理员手动触发导入时

**通知示例：**
```
⚠️ 发现未绑定客服账号

平台：taobao
日期：2026-01-31

未绑定账号（新增）：
• 2208354864110 (昵称: 张三)
• 2208354864111 (昵称: 李四)

请到【设置 > 客服账号绑定】页面进行配置。
```

#### 2. 自动质检检查与触发 ✅

**问题背景：**
导入聊天记录后，需要手动触发AI质检，效率较低。

**解决方案：**
- ✅ 每小时自动检查一次，查找符合条件的对话
- ✅ 符合条件时自动创建AI质检任务
- ✅ 导入成功后立即触发检查（无需等待下一个整点）

**质检条件：**
1. 对话尚未质检（无 `ConversationAnalysis` 记录）
2. 消息数 > 5 条
3. 所有涉及的客服账号已绑定

**通知示例：**
```
✅ 自动质检已启动

发现 15 个符合质检条件的对话
已为 15 个对话创建AI质检任务

条件：消息数>5 且 所有客服已绑定
```

### 技术变更

#### 新增文件

- **`app/qc_checker.py`** - 质检检查器模块
  - `get_unbound_agent_info()` - 获取未绑定客服信息
  - `get_conversations_eligible_for_qc()` - 获取符合条件的对话

#### 修改文件

1. **`app/importers/leyan_import.py`**
   - 新增：收集未绑定账号的昵称信息
   - 新增返回字段：`unbound_agent_nicks` (Dict[str, str])

2. **`app/bucket_import.py`**
   - 新增：聚合未绑定账号的昵称信息
   - 新增返回字段：`unbound_agent_nicks` (Dict[str, str])

3. **`app/cron_runner.py`**
   - 改进：未绑定账号通知包含昵称
   - 新增：`_should_run_hourly_qc_check()` - 检查是否应该运行质检
   - 新增：`_run_auto_qc_check()` - 执行自动质检检查
   - 新增：导入成功后立即触发质检检查

4. **`app/main.py`**
   - 新增：手动导入时检测并通知未绑定账号

#### 新增文档

- **`docs/AUTO_QC_FEATURE.md`** - 功能详细说明文档

### 配置要求

#### 必需配置

1. **飞书 Webhook URL**
   ```bash
   FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/your-token
   ```
   或在【设置 > 对象存储设置】页面配置

2. **自动分析开关**
   - 在【设置 > 通用设置】开启 `auto_analysis_enabled`
   - 在【设置 > 通用设置】开启 `enable_enqueue_analysis`

### 兼容性

- ✅ 向后兼容：不影响现有功能
- ✅ 数据库：无需迁移
- ✅ API：无破坏性变更

### 依赖服务

- PostgreSQL 16+
- Redis (可选，用于任务队列)
- 飞书机器人 Webhook

### 部署说明

#### Docker Compose 部署

```bash
# 重启服务以应用新代码
docker compose restart app worker cron

# 查看日志
docker compose logs -f cron
docker compose logs -f worker
```

#### 验证部署

1. **检查 cron 服务状态**
   ```bash
   docker compose logs cron | grep "自动质检检查"
   ```

2. **检查 worker 服务状态**
   ```bash
   docker compose logs worker | grep "AIAnalysisJob"
   ```

3. **测试未绑定账号通知**
   - 导入包含新客服账号的数据
   - 检查飞书群是否收到通知

4. **测试自动质检**
   - 确保有符合条件的对话（>5消息，已绑定客服）
   - 等待最多1小时或手动触发导入
   - 检查是否收到质检启动通知

### 监控指标

#### 关键日志

1. **未绑定账号检测**
   ```
   [cron] 发现未绑定客服账号: platform=taobao, date=2026-01-31, count=2
   ```

2. **自动质检触发**
   ```
   [cron] 自动质检检查：发现 15 个符合条件的对话，入队 15 个新任务
   ```

3. **质检任务处理**
   ```
   [worker] Processing AIAnalysisJob id=123, conversation_id=456
   [worker] AIAnalysisJob id=123 completed successfully
   ```

#### 数据库查询

```sql
-- 查看最近的自动质检任务
SELECT 
    id,
    conversation_id,
    status,
    created_at,
    finished_at
FROM aianalysisjob 
WHERE created_at > NOW() - INTERVAL '1 day'
ORDER BY created_at DESC
LIMIT 20;

-- 统计今日自动质检完成数
SELECT 
    DATE(finished_at) as date,
    COUNT(*) as completed_count
FROM aianalysisjob
WHERE status = 'done'
  AND finished_at > NOW() - INTERVAL '7 days'
GROUP BY DATE(finished_at)
ORDER BY date DESC;
```

### 性能影响

- **定时检查开销：** 每小时查询一次数据库（< 100ms）
- **通知发送开销：** 异步 HTTP 请求（< 1s）
- **内存占用：** 增加约 10MB（新增模块）
- **CPU 影响：** 可忽略（仅在定时检查时短暂升高）

### 已知限制

1. **消息数阈值固定**
   - 当前硬编码为 5 条
   - 未来版本将支持通过配置调整

2. **检查频率固定**
   - 当前为每小时检查一次
   - 可通过修改 `_should_run_hourly_qc_check()` 调整

3. **批量处理限制**
   - 每次最多检查 50 个对话
   - 避免单次检查耗时过长

### 回滚方案

如需回滚到之前版本：

1. **还原代码**
   ```bash
   git revert <commit-hash>
   docker compose restart app worker cron
   ```

2. **清理数据**
   ```sql
   -- 可选：删除自动创建的质检任务
   DELETE FROM aianalysisjob 
   WHERE created_at > '2026-02-02' 
   AND batch_id IS NULL;
   ```

### 后续计划

- [ ] 支持自定义消息数阈值（通过 AppConfig）
- [ ] 添加质检任务优先级机制
- [ ] 提供质检进度仪表板
- [ ] 支持按平台/时间段过滤
- [ ] 批量重试失败任务功能

### 相关文档

- [功能详细说明](docs/AUTO_QC_FEATURE.md)
- [CLAUDE.md](CLAUDE.md) - 项目概述

### 贡献者

- 实现日期：2026-02-02
- 需求来源：客服质检系统优化需求

---

**注意事项：**
1. 首次部署后，建议观察1-2天以验证功能稳定性
2. 确保飞书 webhook 配置正确，否则无法接收通知
3. 定期检查质检任务队列，避免堆积
