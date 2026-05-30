# MinerU 使用说明

本方案依赖 MinerU 将 PDF 解析为 Markdown 结构，核心证据源是每篇文档的 `combined.md`。

建议目录结构：

```text
combined/
  1/1_combined.md
  2/2_combined.md
  ...
```

V4/V5 均只信任 `combined.md` 中可定位的证据片段。最终 hard gate 会重新在 `combined.md` 中查找 `evidence_quote`，避免模型生成不可追溯文本。

公开仓库不包含完整 `combined.md`，因为其内容仍受原始论文和赛事数据条款约束。
