# 客服分析报表功能更新日志

## 概述

新增"客服日期、场景、满意度交叉分析报表"功能，提供多维度的客服数据分析能力。

## 更新时间

2026-02-02

## 功能清单

### 1. 后端实现

#### 新增路由

**页面路由**：`GET /reports/agent-analysis`
- 功能：渲染客服分析报表页面
- 权限：所有用户（客服限定查看自己）
- 参数：
  - `group_by`: 分组方式（day/week/month）
  - `start`: 开始日期（YYYY-MM-DD）
  - `end`: 结束日期（YYYY-MM-DD）
  - `platform`: 平台筛选（可选）
  - `agent_user_id`: 客服筛选（可选）

**API 路由**：`GET /api/reports/agent-analysis/conversations`
- 功能：获取指定条件的对话列表（CID）
- 权限：所有用户（客服限定查看自己）
- 参数：
  - `agent_id`: 客服 ID
  - `period`: 日期（YYYY-MM-DD）
  - `scenario`: 场景
  - `satisfaction`: 满意度
  - `platform`: 平台（可选）

#### 核心逻辑

1. **多维度统计**
   - 客服维度：基于 `agent_user_id`（优先使用 AgentBinding）
   - 时间维度：使用 PostgreSQL `date_trunc` 函数
   - 场景维度：基于 `ConversationAnalysis.reception_scenario`
   - 满意度维度：基于 `ConversationAnalysis.satisfaction_change`

2. **权限控制**
   - 管理员/主管：查看全部
   - 客服：仅查看自己的数据

3. **数据查询优化**
   - 仅统计最新的 AI 质检结果
   - 支持实时客服绑定映射
   - 分组统计避免大量数据传输

### 2. 前端实现

#### 新增模板

**模板文件**：`app/templates/agent_analysis.html`

**功能特性**：
1. **筛选表单**
   - 分组方式选择（按天/按周/按月）
   - 日期范围选择
   - 平台筛选
   - 客服筛选

2. **多级展开表格**
   - Level 0: 客服汇总（默认显示）
   - Level 1: 日期明细（可展开）
   - Level 2: 场景明细（可展开）
   - Level 3: 满意度明细（可展开，可点击查看对话）

3. **交互功能**
   - 点击展开/折叠图标控制子级显示
   - 点击满意度标签弹窗显示对话列表
   - 点击 CID 快速预览对话（复用全局 CID 预览模态框）

4. **样式设计**
   - 响应式布局，适配不同屏幕尺寸
   - 清晰的视觉层级（缩进显示层级关系）
   - 统一的交互风格（与标签报表保持一致）

#### 导航更新

**修改文件**：`app/templates/base.html`

**更新内容**：
- 主导航栏新增"客服分析"入口
- 链接到 `/reports/agent-analysis`
- 支持当前页面高亮显示

### 3. 文档

#### 新增文档

**文档文件**：`docs/AGENT_ANALYSIS_REPORT.md`

**文档内容**：
- 功能概述
- 访问路径和权限说明
- 功能特点详解
- 使用场景示例
- 操作指南
- 数据说明（统计口径、分类标准）
- 技术实现细节
- 常见问题解答
- 后续优化方向

## 代码变更

### 修改的文件

1. **app/main.py**
   - 新增函数：`agent_analysis_report_page()`
   - 新增函数：`get_agent_analysis_conversations()`
   - 插入位置：标签报表相关路由之后（第 5162 行之后）

2. **app/templates/base.html**
   - 导航栏新增"客服分析"链接
   - 位置：标签报表链接之后

### 新增的文件

1. **app/templates/agent_analysis.html**
   - 客服分析报表页面模板
   - 约 220 行代码

2. **docs/AGENT_ANALYSIS_REPORT.md**
   - 功能文档
   - 约 200 行内容

3. **CHANGELOG_AGENT_ANALYSIS.md**
   - 本文件，更新日志

## 技术细节

### 数据库查询

```sql
-- 示例：按天分组统计
SELECT 
  COALESCE(agent_binding.user_id, conversation.agent_user_id) as agent_user_id,
  DATE_TRUNC('day', COALESCE(conversation.ended_at, conversation.started_at, conversation.uploaded_at)) as period,
  conversation_analysis.reception_scenario as scenario,
  conversation_analysis.satisfaction_change as satisfaction,
  COUNT(DISTINCT conversation.id) as cnt
FROM conversation
JOIN conversation_analysis ON conversation_analysis.conversation_id = conversation.id
JOIN (
  SELECT MAX(id) as analysis_id
  FROM conversation_analysis
  GROUP BY conversation_id
) latest ON latest.analysis_id = conversation_analysis.id
LEFT JOIN agent_binding ON 
  agent_binding.platform = conversation.platform AND 
  agent_binding.agent_account = conversation.agent_account
WHERE ...
GROUP BY agent_user_id, period, scenario, satisfaction
ORDER BY agent_user_id, period, scenario, satisfaction;
```

### 前端多级展开逻辑

```javascript
function toggleGroup(groupId) {
  const ico = document.getElementById('ico-' + groupId);
  const isExpanded = ico && ico.textContent === '▾';
  
  if (isExpanded) {
    // 折叠：隐藏所有子元素和后代
    hideAllDescendants(groupId);
  } else {
    // 展开：显示直接子元素
    showDirectChildren(groupId);
  }
}
```

## 使用示例

### 示例 1：查看某个客服最近一周的表现

1. 访问 `/reports/agent-analysis`
2. 选择"按天"分组
3. 设置日期范围：最近7天
4. 选择客服：张三
5. 点击"查询"
6. 展开客服名称，查看每天的数据
7. 进一步展开查看各场景和满意度分布

### 示例 2：分析售前场景的满意度趋势

1. 访问 `/reports/agent-analysis`
2. 选择"按周"分组
3. 设置日期范围：最近一个月
4. 不限定客服（查看全部）
5. 点击"查询"
6. 逐级展开，关注"售前"场景
7. 对比不同周的满意度分布

### 示例 3：追踪满意度下降的具体对话

1. 在报表中找到"大幅减少"的满意度明细
2. 点击"大幅减少"标签
3. 弹窗显示该条件下的所有对话 CID
4. 点击任意 CID 快速预览对话
5. 进入对话详情页深入分析

## 测试建议

### 功能测试

1. **权限测试**
   - 管理员登录：验证可查看所有客服
   - 主管登录：验证可查看所有客服
   - 客服登录：验证仅可查看自己

2. **筛选测试**
   - 按天/周/月分组测试
   - 日期范围测试（包括边界情况）
   - 平台筛选测试
   - 客服筛选测试

3. **展开/折叠测试**
   - 验证多级展开逻辑
   - 验证折叠时子元素全部隐藏
   - 验证图标切换正确

4. **对话列表测试**
   - 点击满意度标签弹窗显示
   - 验证 API 返回正确的 CID 列表
   - 验证 CID 点击跳转功能

### 性能测试

1. **大数据量测试**
   - 测试 1000+ 对话的查询性能
   - 测试 10+ 客服的展示性能
   - 测试长时间范围（如一年）的查询

2. **并发测试**
   - 测试多用户同时访问
   - 测试多次快速展开/折叠

### 兼容性测试

1. **浏览器兼容性**
   - Chrome（推荐）
   - Safari
   - Firefox
   - Edge

2. **屏幕尺寸**
   - 桌面端（1920x1080）
   - 笔记本（1366x768）
   - 平板（横屏）

## 已知限制

1. **数据范围**
   - 仅统计有 AI 质检结果的对话
   - 场景和满意度为空的对话显示为"未知"

2. **时间分组**
   - 按周分组时，周一为一周的开始（PostgreSQL 默认）
   - 时区使用 UTC（与系统其他部分一致）

3. **展示限制**
   - 暂不支持导出功能
   - 暂不支持可视化图表
   - 暂不支持自定义维度

## 后续优化计划

### 近期优化（1-2周）

1. **数据导出**
   - 支持导出为 Excel 格式
   - 包含所有筛选条件和统计结果

2. **性能优化**
   - 添加查询结果缓存
   - 优化大数据量查询

### 中期优化（1-2个月）

1. **可视化图表**
   - 添加柱状图展示对话数趋势
   - 添加饼图展示场景/满意度分布
   - 支持图表与表格切换

2. **对比分析**
   - 支持多个客服的横向对比
   - 支持时间段的纵向对比
   - 添加环比/同比功能

### 长期优化（3个月+）

1. **高级分析**
   - 支持自定义维度组合
   - 添加预测分析功能
   - 支持异常检测和告警

2. **自动化报告**
   - 支持定时生成报表
   - 支持邮件/飞书推送
   - 支持订阅和分享

## 相关文档

- 功能文档：`docs/AGENT_ANALYSIS_REPORT.md`
- 项目 README：`README.md`
- 开发指南：`CLAUDE.md`

## 联系方式

如有问题或建议，请联系开发团队。
