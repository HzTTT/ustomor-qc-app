# 客服分析报表测试指南

## 快速测试

### 前置条件

1. 系统已启动（`docker compose up`）
2. 已有导入的对话数据且已完成 AI 质检
3. 已有客服绑定记录

### 测试步骤

#### 1. 访问报表页面

```
URL: http://localhost:8000/reports/agent-analysis
```

**预期结果**：
- 页面正常加载
- 显示筛选表单（分组方式、日期范围、平台、客服）
- 显示"查询"按钮

#### 2. 测试默认查询

**操作**：
- 直接点击"查询"按钮（不修改任何筛选条件）

**预期结果**：
- 默认查询最近 7 天的数据
- 默认按天分组
- 如果有数据，显示多级树形表格
- 如果无数据，显示"暂无数据"提示

#### 3. 测试多级展开

**操作**：
1. 点击客服名称前的 `▸` 图标
2. 展开后点击日期前的 `▸` 图标
3. 展开后点击场景前的 `▸` 图标

**预期结果**：
- 第一次点击：图标变为 `▾`，显示下一级数据
- 再次点击：图标变回 `▸`，隐藏所有子级数据
- 缩进正确显示层级关系
- 汇总数字与明细数字一致

#### 4. 测试对话列表弹窗

**操作**：
1. 展开到满意度明细（Level 3）
2. 点击某个满意度标签（如"小幅增加"）

**预期结果**：
- 弹窗正确显示
- 标题显示对话总数
- 副标题显示筛选条件（场景、满意度、日期）
- 列表显示所有符合条件的 CID
- 点击 CID 可快速预览对话

#### 5. 测试筛选功能

**操作**：
1. 修改分组方式为"按周"
2. 修改日期范围为最近一个月
3. 选择特定平台（如"taobao"）
4. 选择特定客服
5. 点击"查询"

**预期结果**：
- 查询参数正确传递（URL 包含查询参数）
- 数据按周分组显示
- 仅显示选定平台和客服的数据
- 筛选条件在表单中正确回显

#### 6. 测试权限控制（客服角色）

**操作**：
1. 使用客服账号登录
2. 访问 `/reports/agent-analysis`

**预期结果**：
- 页面正常访问
- 客服筛选器自动选中当前客服且不可修改
- 仅显示当前客服的数据

#### 7. 测试空数据情况

**操作**：
1. 设置一个不存在数据的日期范围（如明年）
2. 点击"查询"

**预期结果**：
- 页面正常加载
- 显示"暂无数据（可能是还没有AI分析，或筛选条件过窄）"
- 不显示表格

## 数据验证

### 验证统计准确性

**方法 1：SQL 验证**

```sql
-- 查询某个客服某天的对话数
SELECT COUNT(DISTINCT c.id) 
FROM conversation c
JOIN conversation_analysis ca ON ca.conversation_id = c.id
JOIN (
  SELECT conversation_id, MAX(id) as max_id
  FROM conversation_analysis
  GROUP BY conversation_id
) latest ON ca.id = latest.max_id
LEFT JOIN agent_binding ab ON ab.platform = c.platform AND ab.agent_account = c.agent_account
WHERE COALESCE(ab.user_id, c.agent_user_id) = <客服ID>
  AND DATE_TRUNC('day', COALESCE(c.ended_at, c.started_at, c.uploaded_at)) = '<日期>'
  AND ca.reception_scenario = '<场景>'
  AND ca.satisfaction_change = '<满意度>';
```

**方法 2：对比对话列表**

1. 在报表中点击某个满意度标签，记录对话总数
2. 点击查看对话列表，计数 CID 数量
3. 对比两个数字是否一致

### 验证层级汇总

**操作**：
1. 展开到最底层，手动加总某个场景下所有满意度的对话数
2. 对比该场景的汇总数字

**预期结果**：
- 汇总数字 = 所有明细数字之和

## 性能测试

### 测试大数据量

**场景**：查询包含大量对话的日期范围（如一个月）

**观察指标**：
1. 页面加载时间（应在 3 秒内）
2. 查询响应时间（后端日志）
3. 浏览器内存占用

**预期结果**：
- 页面响应流畅，无明显卡顿
- 多级展开操作即时响应

### 测试多客服展示

**场景**：不限定客服，查询所有客服的数据

**观察指标**：
1. 表格渲染时间
2. 展开/折叠操作流畅度

**预期结果**：
- 10 个以内客服：流畅
- 10-50 个客服：可接受
- 50+ 客服：考虑分页或其他优化

## 常见问题排查

### 问题 1：无法访问页面（404）

**可能原因**：
- 路由未正确注册
- 服务未重启

**排查步骤**：
1. 检查 `app/main.py` 中是否有 `@app.get("/reports/agent-analysis")` 路由
2. 重启服务：`docker compose restart app`
3. 查看服务日志：`docker compose logs app`

### 问题 2：显示"暂无数据"

**可能原因**：
- 对话未进行 AI 质检
- 客服未绑定
- 筛选条件过窄

**排查步骤**：
1. 查询是否有 AI 质检结果：
   ```sql
   SELECT COUNT(*) FROM conversation_analysis;
   ```
2. 查询是否有客服绑定：
   ```sql
   SELECT * FROM agent_binding;
   ```
3. 放宽筛选条件（扩大日期范围，不限定平台和客服）

### 问题 3：统计数字不准确

**可能原因**：
- 未使用最新的 AI 质检结果
- 客服绑定错误

**排查步骤**：
1. 验证是否仅统计最新分析：
   ```sql
   SELECT conversation_id, COUNT(*) 
   FROM conversation_analysis 
   GROUP BY conversation_id 
   HAVING COUNT(*) > 1 
   LIMIT 10;
   ```
2. 检查客服绑定表：
   ```sql
   SELECT * FROM agent_binding WHERE platform = '<平台>' AND agent_account = '<账号>';
   ```

### 问题 4：多级展开不工作

**可能原因**：
- JavaScript 错误
- 数据结构问题

**排查步骤**：
1. 打开浏览器开发者工具（F12）
2. 查看 Console 是否有错误
3. 检查 Network 请求是否正常
4. 验证 HTML 的 `data-*` 属性是否正确

### 问题 5：对话列表弹窗无数据

**可能原因**：
- API 返回错误
- 筛选条件不匹配

**排查步骤**：
1. 打开浏览器开发者工具 Network
2. 查看 API 请求 `/api/reports/agent-analysis/conversations`
3. 检查请求参数和响应内容
4. 验证数据库中是否有符合条件的对话

## 自动化测试建议

### 单元测试

```python
# tests/test_agent_analysis.py

def test_agent_analysis_page_accessible(client, auth_admin):
    """测试页面可访问"""
    response = client.get("/reports/agent-analysis")
    assert response.status_code == 200

def test_agent_analysis_with_filters(client, auth_admin):
    """测试带筛选条件的查询"""
    response = client.get("/reports/agent-analysis?group_by=week&start=2026-01-01&end=2026-01-31")
    assert response.status_code == 200
    assert "按周" in response.text

def test_agent_analysis_api(client, auth_admin):
    """测试 API 返回"""
    response = client.get("/api/reports/agent-analysis/conversations?period=2026-01-20&scenario=售前&satisfaction=小幅增加")
    assert response.status_code == 200
    data = response.json()
    assert "cids" in data
    assert isinstance(data["cids"], list)
```

### 集成测试

```python
def test_agent_analysis_workflow(client, auth_admin, sample_conversations):
    """测试完整工作流"""
    # 1. 访问页面
    response = client.get("/reports/agent-analysis")
    assert response.status_code == 200
    
    # 2. 查询数据（应有结果）
    response = client.get("/reports/agent-analysis?start=2026-01-01&end=2026-01-31")
    assert "暂无数据" not in response.text
    
    # 3. 获取对话列表
    response = client.get("/api/reports/agent-analysis/conversations?period=2026-01-20&scenario=售前&satisfaction=小幅增加")
    data = response.json()
    assert len(data["cids"]) > 0
```

## 测试清单

- [ ] 页面正常加载
- [ ] 默认查询有结果
- [ ] 多级展开/折叠工作正常
- [ ] 对话列表弹窗显示正常
- [ ] 筛选功能正常（分组方式、日期、平台、客服）
- [ ] 权限控制正常（客服只看自己）
- [ ] 空数据情况处理正常
- [ ] 统计数字准确（对比 SQL 查询）
- [ ] 层级汇总准确（明细之和 = 汇总）
- [ ] 性能可接受（大数据量、多客服）
- [ ] CID 点击预览功能正常
- [ ] 响应式布局正常（不同屏幕尺寸）

## 反馈

测试过程中发现的问题请记录：
1. 问题描述
2. 重现步骤
3. 预期结果 vs 实际结果
4. 浏览器和操作系统版本
5. 相关截图或日志
