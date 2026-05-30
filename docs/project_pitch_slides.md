# MinerU 材料文献智能解析 Agentic Parser V1 - PPT 文案

## 1. 任务与痛点
- 材料实体命名复杂，属性分散于正文、表格和图注。
- 少样本、长文档和跨章节绑定是主要难点。

## 2. 核心方案
- V4 AI reader 做文档级宽召回。
- V5 semantic binder 做材料-属性-数值-证据绑定。
- Deterministic gate 做 quote/schema/unit/duplicate 校验。

## 3. 数据流
- MinerU combined.md -> V4 candidates -> retry/merge -> V5 semantic bind -> hard gate -> dataset.jsonl。

## 4. 本次结果
- 输入候选 1421 条。
- 最终高置信 688 条。
- hard gate 错误 0。
- 覆盖 90 篇文档。

## 5. 创新点
- AI 做语义阅读，代码做确定性证据约束。
- 用 measured entity / source property label 降低对象错绑。
- 保守丢弃 review/book chapter，提高提交精度。

## 6. 应用价值
- 构建材料事实知识库。
- 支持工艺参数和性能指标检索。
- 可扩展到更多材料属性和领域 schema。
