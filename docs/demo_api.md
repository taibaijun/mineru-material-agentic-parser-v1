# Demo/API 使用说明

本项目提供一个标准库 HTTP Demo，无需安装 FastAPI。

启动：

```powershell
python demo/material_api.py --dataset dataset.jsonl --host 127.0.0.1 --port 8765
```

接口：

- `/health`：健康检查。
- `/stats`：数据统计。
- `/records?property=process_temperature&limit=20`：按属性过滤。
- `/records?doc_id=3&limit=20`：按文档过滤。
- `/search?q=sintered&limit=10`：在材料、属性、证据、条件中检索。

该 Demo 用于验证数据可读、可检索、可集成；完整批处理复现见 `scripts/run_v4_v5_pipeline.ps1`。
