# 数据字段说明

每行 JSON 表示一条材料属性事实。

核心字段：

- `doc_id`：文档编号。
- `material`：材料实体或样品名。
- `property`：目标属性，当前包括 `process_temperature`、`thermal_transition`、`discharge_capacity`、`ionic_conductivity`、`adhesion_strength`。
- `property_subtype`：属性细分，例如 sintering temperature、glass transition temperature。
- `value` / `value_max`：数值或区间上界。
- `value_text`：原始数值文本。
- `unit`：标准化单位。
- `condition`：温度、时间、气氛、倍率等条件。
- `evidence_quote`：来自 MinerU `combined.md` 的证据片段。
- `source_property_label`：原文中的属性标签或表头。
- `target_property_match`：V5 判断是否匹配目标属性。
- `material_binding`：measured entity 和角色绑定说明。
- `evidence_assessment`：证据是否直接支持该事实。
- `source_doc_type`：来源类型，最终数据只保留 `primary_research` 和 `patent`。
- `confidence`：模型置信度。
- `record_id`：稳定记录 ID。

正式提交建议使用根目录 `dataset.jsonl`。
