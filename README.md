# 制造设备故障知识库问答 Agent（RAG）

> 面向制造设备故障场景的垂直领域 RAG 知识库问答系统。核心亮点是**自研混合检索**（向量检索 + BM25 关键词检索 + RRF 融合 + rerank 精排），并用召回率对比实验证明其优于纯向量检索。

---

## 一、项目定位

- **场景**：制造设备故障知识库问答（覆盖注塑机、数控机床、空压机、电机、PLC、变频器、液压、气动、机器人、传感器等 20+ 类设备）
- **核心改进点**：混合检索（Hybrid Retrieval），自己实现，不依赖 LangChain 等重框架
- **回答带引用**：答案附上引用的原文片段与出处，可溯源

## 二、核心亮点

- **自研混合检索**：向量检索（FAISS + bge）+ BM25 关键词检索（jieba 分词）+ RRF 融合 + bge-reranker 精排
- **对比实验证明有效**：在 50 题测试集上，混合检索的 **MRR 提升 28.6%**（0.301→0.387），**Recall@1 从 0% 提升到 20%**
- **完整 RAG 链路自己实现**：文档切分、embedding 建库、检索、生成、引用解析，每一步都能讲清原理
- **生产级工程细节**：统一 LLM 接入层（OpenAI 兼容，可换 DeepSeek/通义/智谱）、本地模型推理（不烧 API 钱）、异常处理

## 三、技术架构

```
用户提问
   │
   ▼
┌─────────────────────────────────────────┐
│  检索模块（改进点核心）                   │
│  ① 向量检索  FAISS + bge-large-zh       │
│  ② 关键词检索 BM25 + jieba               │
│  ③ RRF 融合去重                          │
│  ④ bge-reranker 精排                     │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  生成模块：拼接 prompt → DeepSeek        │
│  返回答案 + 引用来源                     │
└─────────────────────────────────────────┘
```

## 四、技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| LLM | DeepSeek（OpenAI 兼容） | 统一接入层，可换通义/智谱 |
| Embedding | bge-large-zh-v1.5 | 中文开源 SOTA，本地 CPU 推理，免费 |
| 向量库 | FAISS | 工业界最常用，轻量 |
| 关键词检索 | rank_bm25 + jieba | 经典 BM25，中文分词 |
| 重排序 | bge-reranker-base | 召回 + 精排两阶段 |
| 界面 | CLI + Gradio Web | 命令行 + 网页双 demo |

## 五、目录结构

```
rag/
├── config.py            # 全局配置（路径、检索参数、LLM 接入）
├── .env.example         # 环境变量模板（真实 .env 已 gitignore）
├── requirements.txt     # 依赖清单
├── download_models.py   # 下载 bge 模型（ModelScope 镜像）
├── build_index.py       # 建库：切分 → embedding → FAISS + BM25
├── cli.py               # 命令行问答（支持交互/单次）
├── web.py               # Gradio Web 界面
├── evaluate.py          # 召回率对比实验
├── rag/                 # 核心包（自己实现）
│   ├── loader.py        # 文档读取 + 章节切分
│   ├── indexer.py       # embedding + FAISS + BM25 建索引
│   ├── retriever.py     # 向量检索 + BM25 + RRF + rerank
│   ├── generator.py     # prompt 拼接 + LLM + 引用解析
│   └── llm.py           # 统一 LLM 接入层
├── data/
│   ├── docs/            # 50 篇制造设备故障文档
│   └── testset/         # 50 题测试集（问题 + 标准答案文档）
├── tests/               # 单元测试
├── models/              # 下载的模型权重（gitignore）
└── index_store/         # 建库产物（gitignore）
```

## 六、快速开始

### 1. 环境要求

- Python 3.9+（本项目在 3.13 上开发）
- 无需 GPU（全程 CPU 推理）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 LLM Key

```bash
cp .env.example .env   # 填入 DeepSeek API Key
```

`.env` 采用 OpenAI 兼容协议，`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 三项即可切换 DeepSeek / 通义千问 / 智谱。

### 4. 下载模型（约 2GB，走 ModelScope 国内镜像）

```bash
python download_models.py
```

### 5. 建库

```bash
python build_index.py
```

### 6. 问答

```bash
# 命令行（单次）
python cli.py "注塑机飞边怎么排查处理"

# 命令行（交互）
python cli.py

# Web 界面（浏览器打开 http://127.0.0.1:7860）
python web.py
```

## 七、混合检索原理（面试重点）

**为什么纯向量检索不够？** 向量检索懂语义（"电池鼓包"能联想到"膨胀"），但会漏掉文档里恰好有精确词的段落；BM25 精确命中关键词，但不懂同义词。两者互补。

**混合检索流程**：

| 步骤 | 做什么 | 说明 |
|---|---|---|
| ① 向量检索 | bge 把问题变成向量，FAISS 找最相似 top-20 | 找"意思相近"的 |
| ② 关键词检索 | BM25 按关键词匹配 top-20 | 找"有精确关键词"的 |
| ③ RRF 融合 | 对两路结果做 Reciprocal Rank Fusion（按排名倒数加权）去重 | 合并排序 |
| ④ rerank 精排 | bge-reranker 对融合结果重排，取最终 top-5 | 精挑一遍 |
| ⑤ 生成 | top-5 拼进 prompt 给 LLM，答案带引用编号 | 照着资料答 |

## 八、对比实验（真实数据，可复现）

```bash
python evaluate.py
```

评估口径：**chunk 级**（比文档级更严格），检索前 k 个文本块中，命中"标准答案所在章节"（排查步骤/常见原因）才算召回成功。测试集 50 题。

| 指标 | 纯向量检索 | 混合检索 | 提升 |
|---|---|---|---|
| Recall@1 | 0.00% | 20.00% | **+20.00%** |
| Recall@3 | 54.00% | 58.00% | +4.00% |
| Recall@5 | 72.00% | 68.00% | -4.00% |
| **MRR** | 0.3010 | 0.3870 | **+28.6%** |

**结论解读**：

- **排序更准**：MRR 提升 28.6%、Recall@1 提升 20%，说明混合检索把"标准答案段落"排到了更靠前的位置。纯向量检索的 top1 往往是"标题/现象"块（语义最相似），而非真正回答问题的"排查/原因"段落——这正是 rerank 要解决的问题。
- **Recall@5 接近**：在宽松口径（前 5 个）下两者接近，原因是知识库规模有限（50 篇、250 块）、bge 向量检索本身已较强。混合检索的收益在更大规模、更相似的语料上会更明显。

## 九、单元测试

```bash
python -m pytest tests/ -q
```

6 个测试覆盖：章节切分、Markdown 清理、jieba 分词、RRF 融合去重、prompt 拼接、引用提取。

## 十、面试讲法（30 秒）

> "我做了一个垂直领域的 RAG 知识库问答系统。核心亮点是我**自己实现了混合检索**——把向量检索和 BM25 关键词检索融合，再加 rerank 精排。因为纯向量检索有个问题：它懂语义，但 top1 往往命中的是故障的『现象描述』，而不是用户真正要的『排查步骤』。我做了一组对比实验，混合检索把标准答案段落的首位命中率 Recall@1 从 0% 提到了 20%，MRR 提升了 28.6%。整个链路从文档切分、embedding、建库到生成都是自己写的，没用 LangChain 套。"
