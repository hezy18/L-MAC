# L-MAC

L-MAP 是一个基于 LLM 的特归因生成框架。包含特征选择方法（如 Lasso、IG、mRMR）的实现。L-MAC 可用于：
* 特征归因与聚合
* 归因引导预测
* 模型评估
  
本仓库包含完整的模块化实现，包括：mapping/、merging/、rule_generation/、evaluation/ 等核心模块，以及 baselines/ 中的多种传统方法。

## 项目结构
```bash
L-MAP-main/
├── main.py                     # 主入口
├── baselines/                  # IG, Lasso, mRMR 等 baseline 方法
├── llm/                        # 大语言模型、输入输出约束、prompt 构建
├── modules/
│   ├── mapping.py              # 特征映射
│   ├── merging.py              # 特征合并
│   ├── merging_by_region.py    # 按区域合并特征
│   ├── rule_generation.py      # 归因生成
│   ├── inference.py            # 推理模块
│   ├── evaluation.py           # 性能评估
│   └── summary.py              # 结果摘要
├── utils/                      # 工具函数
```

## 运行方式
项目入口为 main.py，可通过命令行指定配置文件或数据集路径。

```bash
python main.py --model 'gpt4o' --setting 'summary' --data_date_version '20250401-30' --pred_num 10
```

需配置大语言模型 API Key，在 llm/llm_client.py 或 config/config.py 中设置
```python
OPENAI_API_KEY = "your-api-key"
```

### L-MAC 会输出：
* LLM 生成的素材级别归因
* 聚合后的全局归因
* 归因引导的新素材表现预测
* 模型性能比较
