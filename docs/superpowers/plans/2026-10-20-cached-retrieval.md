# MAR 检索缓存与 Qwen3.8-27B 实施计划

> 执行方式：按任务分工并行实施，主代理逐项集成和检查。当前会话继续执行，不另行请求执行确认。

**目标：** 在独立目录部署 SearXNG、接入可复用检索缓存，并使用 Qwen3.8-27B 完成可验证的推理路径。

**架构：** 复制 MAR 的轻量代码，引用原目录已下载数据。统一缓存保存搜索响应和证据；SearXNG 提供其实际支持的免费搜索，原 Serper/ImgBB 用于首次补齐真正的反向搜图。Qwen 服务使用独立端口和运行配置。

**技术栈：** Python、SQLite、HTTPX、Docker、SearXNG、OpenAI 兼容接口、原 MAR/LangGraph。

**规格：** `docs/superpowers/specs/2026-10-20-cached-retrieval.md`

## 全局约束

- 原项目不修改；新目录为 `/home/c/czh/MAR_KDD_SearXNG`。
- 两套作者测试集各 1,000 条，输入标签不进入推理或检索。
- 使用官方 `Qwen/Qwen3.8-27B` 或其官方 FP8 权重，实际精度写入运行记录。
- 无缓存、零结果、后端错误必须分开；缓存模式不得联网。
- 密钥不打印、不入版本记录；仅绑定回环地址。

## 任务一：建立独立运行目录

**文件：** 复制 `MAR_framework/`、`reverse_image/`、必要 `scripts/`；创建 `configs/runtime.json`、`run.sh`、`.gitignore`。

- [x] 复制轻量代码，排除 `.git`、模型、输出、虚拟环境和密钥文件。共享数据用显式绝对路径，避免重复占用磁盘。
- [x] 创建独立配置，默认模型名 `Qwen3.8-27B`、SearXNG 地址 `http://127.0.0.1:18888`、模型地址 `http://127.0.0.1:19888/v1`，端口占用时改为实测空闲端口。
- [x] 记录原代码来源与文件哈希；确认原项目状态未被修改。

## 任务二：持久缓存与网页检索适配

**文件：** `retrieval/__init__.py`、`retrieval/cache.py`、`retrieval/search.py`、`MAR_framework/web_search.py`、`tests/test_cache.py`。

**接口：**

```python
class EvidenceCache:
    def __init__(self, path): ...
    def get(self, namespace: str, key: dict): ...
    def put(self, namespace: str, key: dict, value: dict): ...

class SearchClient:
    def __init__(self, *, cache, base_url, mode="online", timeout=20, max_results=10): ...
    def search(self, query: str, category="general") -> tuple[list, list]: ...
```

- [x] 测试第二次查询不访问后端、参数变化生成不同缓存、只读缺失不联网、HTTP 故障不保存为成功零结果。
- [x] 实现 SQLite 原子写入，键由规范化 JSON 的 SHA-256 构成，值保留状态、时间、来源和响应。
- [x] SearXNG 请求 `/search?q=...&format=json&categories=...`；保留 `unresponsive_engines`，对全部引擎失败显式报告。
- [x] 保持原 `WebSearchClient.search()` 的 `(rows, events)` 接口，`rows` 字段为 `title/href/body`，`events` 记录缓存命中。
- [x] 执行 `python -m unittest discover -s tests -v`，核对实际请求计数。

## 任务三：部署 SearXNG

**文件：** `deploy/searxng/settings.yml`、`deploy/searxng/compose.yml`、`scripts/searxng.sh`。

- [x] 核实官方容器与引擎能力，配置 `search.formats: [html, json]`、本机端口、随机实例密钥。
- [x] 启动本项目专属服务，保留镜像摘要；不得停止其他容器。
- [x] 用网页和图片类别分别实测，保存响应摘要；确认图片类别是否只是文本搜图，反向搜索能力按实测标注。

## 任务四：导入和补齐图像证据

**文件：** `scripts/prepare_cached_evidence.py`、`retrieval/reverse.py`、`tests/test_reverse_cache.py`、`data/`。

**接口：** 输入作者 `samples.json`，输出兼容原 `evidence_manifest.attach_evidence` 的 `reverse-evidence.json`。

- [x] 导入作者图注与原目录已有成功记录，校验图片/图注哈希；不导入错误为成功。
- [x] 以图片内容和检索配置生成缓存键，缓存原始候选和最终提取证据；没有图像输入能力的 SearXNG 结果标记为补充文本搜图。
- [x] 首次补齐允许使用已有 Serper/ImgBB 密钥；限制并发、记录错误、遇到配额故障停止收费后端，保留已完成结果。
- [x] 为两套测试数据生成完整状态清单。测试重复图片不重搜、错误可重试、只读缺失不调用上传/搜索、恢复不中断已完成结果。
- [x] 实测新增检索及重复命中；为全量样本启动免费图片候选预缓存，记录逐条状态和断点。
- [ ] 全部原生反向搜图证据补齐：当前 Serper 已返回 `quota_exhausted`，未将免费文字搜图标记为原生反搜。

## 任务五：Qwen3.8-27B 与最终验收

**文件：** `scripts/download_model.py`、`scripts/serve_qwen.sh`、`configs/runtime.json`、`README.md`。

- [x] 依据当前 GPU、磁盘和官方模型卡确定 BF16/官方 FP8 部署，下载到独立目录并固定版本。
- [x] 使用独立服务名称与端口启动，验证 `/v1/models` 返回实际 Qwen3.8-27B。
- [x] 用真实图像请求确认视觉能力，再运行一条三轮 MAR；记录模型、缓存模式和证据覆盖率。
- [x] 运行本次变更相关测试，检查只读缓存模式不触发外部搜索；更新中文使用说明和实测状态。
- [x] 最终交付目录、启动命令、缓存覆盖率、模型状态，以及真实未解决的限制。

## 验收补充

- 作者图注和原项目导入证据冻结保留；生成证据增加输入和配置指纹。
- 原始候选与最终图注分层缓存；`--refresh` 不因重新提取而重复调用收费搜索。
- SearXNG 检索方式传入模型，图注文本及哈希保持原样。
- 报告兼容 DGM4、MMFakeBench 和仅用于流程验证的混合小样本。
- 所有网络故障保留为错误，不计入成功空结果或完整评估。

## 用户追加的全量后台推理

- [x] 两套各 1,000 条，8 路并发、三轮 MAR，固定完整输入与运行快照。
- [x] 独立后台启动，确认 PPID=1、独立会话，8 条样本实际同时调用模型。
- [x] 自动更新 Word 文档、逐条结果、按数据集的论文指标、吞吐及剩余时间。
- [x] 断点恢复与两轮错误重试；缺失图像证据明确记录，网页检索全失败不算成功。
- [ ] 全量推理完成：后台持续运行，以 `outputs/author2000-qwen38/progress.json` 为准。
