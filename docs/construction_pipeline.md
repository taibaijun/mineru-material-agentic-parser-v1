# 构建流程

## 输入

- MinerU 解析后的 `combined.md` 目录。
- 召回候选 `focused_evidence_candidates.recall_v3.jsonl`。
- 材料赛题 schema 和任务说明。
- `DEEPSEEK_API_KEY` 环境变量。

## 阶段 1：V4 文档级宽召回

`run_agentic_material_corpus_v4.py` 对每篇文献读取候选证据窗口和全文上下文，抽取 doc-level material fact candidates。该阶段目标是召回，不直接作为提交。

## 阶段 2：失败重试与合并

单文档 JSON 失败单独重试。本次 `doc41` 首轮失败，重试后补回 1 条候选。随后合并为 1421 条 V5 输入。

## 阶段 3：V5 语义绑定

`run_agentic_material_corpus_v5.py` 对候选事实做材料实体、属性、数值和证据的语义确认。模型可以 ACCEPT、REVISE 或 REJECT；REVISE 后仍会进入本地 hard gate。

## 阶段 4：确定性 gate

本地代码检查字段、schema、单位、quote 对齐、材料绑定、属性绑定和重复项。最终输出 `dataset.jsonl`。

## 阶段 5：提交包

生成质量报告、字段说明、Demo/API、PPT、manifest 和 checksums。公开仓库不包含原始 PDF 和全文解析文件。
