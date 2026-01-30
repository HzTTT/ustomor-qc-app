# Overlay Zip（直接覆盖用）

修复两处页面 500：
1) /conversations/{id} 详情页：NameError / 模板需要 display_agent_by_conv
2) /conversations 列表页：模板需要 display_agent_by_conv，但后端未传入

## 使用方法
解压到项目根目录（与 docker-compose.yml 同级），允许覆盖同名文件即可。

## 覆盖文件
- app/main.py


## 2026-01-27 更新：AI日报后台任务
- 生成日报改为入队后台任务（刷新页面不会丢）。
- worker 会自动处理 DailyAISummaryJob 并写入 DailyAISummaryReport。
- 默认 OPENAI_TIMEOUT 提升到 900 秒（可在 .env 调整）。
