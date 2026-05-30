# MinerU 材料文献智能解析 Agentic Parser V1

这是面向 **材料赛题：基于 MinerU 的材料文献智能解析应用** 的提交包和开源代码。系统以 MinerU 解析后的 `combined.md` 为证据源，使用 AI 论文阅读 agent 做宽召回和语义绑定，再用确定性 gate 做证据、schema、单位和属性一致性校验，输出可追溯的材料实体属性数据。

## 评审快速入口

- 主数据文件：`dataset.jsonl`，共 **688** 条高置信材料属性记录。
- 数据副本：`data/submission_candidates.jsonl`。
- 质量报告：`quality_report.json`。
- 技术报告：`docs/technical_report.md`。
- 构建流程：`docs/construction_pipeline.md`。
- 字段说明：`docs/schema.md`。
- Demo/API：`demo/material_api.py`，仅依赖 Python 标准库。
- 复现脚本：`scripts/run_v4_v5_pipeline.ps1`。
- 验证脚本：`scripts/validate_submission.py`。
- 项目介绍 PPT：`docs/project_pitch_slides.pptx`。
- 开源仓库：`https://github.com/taibaijun/mineru-material-agentic-parser-v1`。

## 当前提交结果

- V4 文档级宽召回：181 个候选文档，doc41 首轮 JSON 失败后已重试补回。
- V5 输入候选：1421 条。
- 最终保留：688 条。
- review/丢弃：733 条。
- 覆盖文档：90 篇有最终记录。
- 本地 hard gate：0 个错误。
- 证据对齐：{'exact': 678, 'whitespace_compact': 10}。
- 最终来源：{'primary_research': 143, 'patent': 545}。

## 数据分布

| 属性 | 记录数 |
| --- | ---: |
| `adhesion_strength` | 55 |
| `discharge_capacity` | 117 |
| `ionic_conductivity` | 77 |
| `process_temperature` | 393 |
| `thermal_transition` | 46 |

## 一键本地查看 Demo

```powershell
python demo/material_api.py --dataset dataset.jsonl --host 127.0.0.1 --port 8765
```

然后访问：

- `http://127.0.0.1:8765/health`
- `http://127.0.0.1:8765/stats`
- `http://127.0.0.1:8765/records?property=ionic_conductivity&limit=10`
- `http://127.0.0.1:8765/search?q=sintered&limit=10`

## 复现方式

完整复现需要评审本地拥有 MinerU 解析结果目录和召回候选文件。设置 `DEEPSEEK_API_KEY` 后按 `scripts/run_v4_v5_pipeline.ps1` 中的路径模板运行。

公开仓库不包含原始论文、PDF 或完整 `combined.md` 全文；这些材料由赛事附件和原始出版许可约束。结构化抽取层按 `LICENSE` 说明开放。
