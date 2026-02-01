# 自动质检功能快速开始指南

本指南帮助你快速配置和测试自动质检功能。

## 前置条件

1. ✅ Docker Compose 环境已启动
2. ✅ 数据库中已有聊天记录
3. ✅ 已创建飞书机器人（可选）

## 5分钟快速配置

### 步骤1: 配置飞书 Webhook（推荐）

1. **创建飞书机器人**
   - 打开飞书，进入需要接收通知的群聊
   - 点击右上角 `···` → `设置` → `群机器人`
   - 点击 `添加机器人` → `自定义机器人`
   - 设置机器人名称（例如：质检通知）
   - 复制 Webhook 地址

2. **配置 Webhook URL**
   
   方式一：通过 Web 界面（推荐）
   ```
   访问: http://localhost:8000/settings/bucket
   找到: 飞书 Webhook URL
   粘贴: 你的 webhook 地址
   点击: 保存
   ```
   
   方式二：通过环境变量
   ```bash
   # 编辑 .env 文件
   FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/your-token
   
   # 重启服务
   docker compose restart app cron
   ```

### 步骤2: 开启自动分析

1. 访问：http://localhost:8000/settings/general
2. 找到：自动分析设置
3. 勾选：
   - ☑️ 启用自动分析
   - ☑️ 自动入队分析任务
4. 点击：保存

### 步骤3: 运行测试

```bash
# 在项目根目录执行
docker compose exec app python /app/scripts/test_auto_qc.py
```

**预期输出：**
```
============================================================
自动质检功能测试
============================================================

=== 数据库统计 ===
对话总数: 150
未质检对话: 45
已绑定客服账号: 3
待处理质检任务: 0
正在处理质检任务: 0

=== 测试1: 未绑定客服账号检测 ===
✓ 找到 5 个客服账号

平台: taobao
  客服账号总数: 5
  ⚠️  未绑定账号: 2
    • 2208354864110 (昵称: 张三)
    • 2208354864111 (昵称: 李四)

=== 测试2: 符合质检条件的对话查找 ===

消息数阈值: >5
  ✓ 找到 10 个符合条件的对话（显示前10个）
    • CID=123 | taobao:xxx:buyer1:2026-01-31 | 15条消息 | 客服: agent1
    • CID=124 | taobao:xxx:buyer2:2026-01-31 | 8条消息 | 客服: agent2
    • CID=125 | taobao:xxx:buyer3:2026-01-31 | 20条消息 | 客服: agent1
    ... 还有 7 个对话

=== 测试3: 飞书通知发送 ===
✓ Webhook URL: https://open.feishu.cn/open-apis/bot/v2/hook...
发送测试通知...
✅ 通知发送成功

============================================================
测试总结
============================================================
未绑定账号检测: ✅ 通过
符合条件对话查找: ✅ 通过
飞书通知发送: ✅ 通过

🎉 所有测试通过！
```

## 功能验证

### 验证1: 未绑定账号通知

**操作：**
```bash
# 手动触发导入（包含未绑定客服的数据）
访问: http://localhost:8000/settings/bucket
选择日期: 2026-01-31
点击: 立即抓取
```

**预期结果：**
- ✅ Web 界面显示导入成功
- ✅ 飞书群收到未绑定账号通知
- ✅ 通知中包含客服昵称

**如果没有收到通知：**
1. 检查 Webhook URL 是否配置正确
2. 检查是否确实有未绑定的账号
3. 查看日志：`docker compose logs cron | grep "未绑定"`

### 验证2: 自动质检触发

**前提条件：**
1. 所有客服账号已绑定
2. 存在 >5 条消息的未质检对话

**操作：**
```bash
# 手动触发导入（会立即检查质检条件）
访问: http://localhost:8000/settings/bucket
选择日期: 2026-01-31
点击: 立即抓取
```

**预期结果（如果有符合条件的对话）：**
- ✅ Web 界面显示导入成功
- ✅ 飞书群收到自动质检启动通知
- ✅ 质检任务开始处理

**验证任务状态：**
```bash
# 查看待处理任务
docker compose exec db psql -U qc -d qc -c "SELECT id, conversation_id, status, created_at FROM aianalysisjob WHERE status='pending' LIMIT 5;"

# 查看 worker 日志
docker compose logs worker | tail -20
```

**如果没有触发质检：**
1. 确认自动分析开关已开启
2. 确认存在符合条件的对话（>5消息 且 客服已绑定）
3. 查看日志：`docker compose logs cron | grep "自动质检"`

### 验证3: 每小时自动检查

**说明：**
- 定时任务每小时自动检查一次
- 导入成功后也会立即检查

**查看检查记录：**
```bash
# 查看最近的检查日志
docker compose logs cron | grep "自动质检检查"

# 示例输出：
# [2026-02-02 14:00:05] 自动质检检查：发现 10 个符合条件的对话，入队 10 个新任务
# [2026-02-02 15:00:03] 自动质检检查：无符合条件的对话
```

## 绑定客服账号

如果测试显示有未绑定的客服账号：

1. **访问绑定页面**
   ```
   http://localhost:8000/settings/agents
   ```

2. **创建客服账号绑定**
   - 选择平台：淘宝/抖音
   - 输入外部账号：例如 `2208354864110`
   - 输入姓名：例如 `张三`（通知中会显示昵称作为提示）
   - 选择站内用户：选择对应的客服用户
   - 点击：创建绑定

3. **验证绑定**
   ```bash
   # 重新运行测试
   docker compose exec app python /app/scripts/test_auto_qc.py
   
   # 应该显示该账号已绑定
   ```

## 创建测试数据（可选）

如果你的数据库中没有足够的测试数据：

```bash
# 生成测试 JSON 数据
docker compose exec app python /app/scripts/generate_fake_json.py

# 导入测试数据
# 方法1: 通过 Web 界面上传 JSON 文件
# 方法2: 使用 bucket 导入功能
```

## 常见问题

### Q1: 测试脚本报错 "No module named 'qc_checker'"

**解决：**
```bash
# 确保在容器内运行
docker compose exec app python /app/scripts/test_auto_qc.py

# 或者在本地运行（需要先安装依赖）
cd app
pip install -r requirements.txt
python ../scripts/test_auto_qc.py
```

### Q2: 飞书通知发送失败

**可能原因：**
1. Webhook URL 配置错误
2. 网络连接问题
3. 飞书机器人被禁用

**排查步骤：**
```bash
# 1. 检查配置
docker compose exec app python -c "
from sqlmodel import Session
from db import engine
from notify import get_feishu_webhook_url
with Session(engine) as s:
    print(get_feishu_webhook_url(s))
"

# 2. 手动测试 webhook
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"msg_type":"text","content":{"text":"测试消息"}}'
```

### Q3: 自动质检没有触发

**检查清单：**
- [ ] 自动分析开关已开启
- [ ] 存在未质检的对话（>5消息）
- [ ] 对话中的客服账号都已绑定
- [ ] worker 服务正常运行
- [ ] cron 服务正常运行

**诊断命令：**
```bash
# 检查服务状态
docker compose ps

# 检查符合条件的对话
docker compose exec db psql -U qc -d qc -c "
SELECT c.id, c.external_id, COUNT(m.id) as msg_count
FROM conversation c
LEFT JOIN message m ON m.conversation_id = c.id
LEFT JOIN conversationanalysis ca ON ca.conversation_id = c.id
WHERE ca.id IS NULL
GROUP BY c.id
HAVING COUNT(m.id) > 5
LIMIT 10;
"

# 检查客服绑定状态
docker compose exec db psql -U qc -d qc -c "
SELECT platform, agent_account, user_id 
FROM agentbinding 
LIMIT 10;
"
```

## 下一步

功能验证成功后，你可以：

1. **调整配置**
   - 修改消息数阈值（默认 5）
   - 调整检查频率（默认 1 小时）

2. **查看详细文档**
   - [功能详细说明](AUTO_QC_FEATURE.md)
   - [更新日志](../CHANGELOG_AUTO_QC.md)

3. **监控运行状态**
   - 查看飞书通知
   - 查看定时任务日志
   - 查看质检任务队列

## 获取帮助

遇到问题？

1. 查看日志：`docker compose logs cron worker`
2. 运行诊断：`docker compose exec app python /app/scripts/test_auto_qc.py`
3. 查看文档：`docs/AUTO_QC_FEATURE.md`

---

**配置完成后，系统将自动：**
- ✅ 每次导入时检测未绑定客服账号并通知
- ✅ 每小时检查一次符合条件的对话
- ✅ 自动创建AI质检任务
- ✅ 通过飞书实时通知重要事件
