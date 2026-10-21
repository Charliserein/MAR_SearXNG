# 本地 Qwen3.8-27B

使用官方 `Qwen/Qwen3.8-27B-FP8` 权重，对外模型名为 `Qwen3.8-27B`。
这是官方 FP8 量化版，文本和图像推理均保留。

- 固定版本：`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。
- 服务地址：`http://127.0.0.1:19888/v1`。
- 独立环境：项目内 `.venv-vllm`，依赖见 `requirements.lock.txt`。
- 权重目录：项目内 `models/Qwen3.8-27B-FP8`，清单 `download-manifest.json`。
- 启动配置：GPU 0、显存预算 0.78、16K 上下文、同时处理最多八个请求、默认关闭 thinking。
- 服务日志与 PID：`outputs/model-server.log`、`outputs/model-server.pid`。

本机真实图像验收已通过：DGM4 第一条图像约 7.7 秒返回合法 JSON，
未向模型发送标签。完整记录见 `outputs/model-vision-smoke.json`，
环境及资源记录见 `outputs/model-runtime.json`。本机 RTX 4090 D 上，
vLLM 使用 Marlin 的 weight-only FP8 路径运行这些官方权重。

四路并行也已实测通过：两套数据各选两张图像同步提交，四个请求均返回合法 JSON，
总耗时 6.031 秒，服务指标确实观测到同时运行四个请求。GPU 0 峰值为
43,593 MiB（包含原有服务），GPU 1 保持 7,887 MiB；记录见
`outputs/model-concurrency-smoke.json`。这是短图像请求的验收时间，
完整 MAR 多轮评测耗时取决于检索与实际输出长度。

随后用同图、同提示、每个请求固定 256 个输出 token 比较吞吐：四路批次为
99.17 tokens/s，八路为 185.97 tokens/s，提升约 87.5%。八路批次共
2,048 个输出 token，耗时 11.012 秒；服务确实同时执行八个请求，KV 缓存峰值
21.3%，无预占或 OOM，GPU 0 显存峰值 43,941 MiB。因此当前保留八路配置，
对比记录为 `outputs/model-throughput-comparison.json`。该结果衡量模型吞吐，
不能直接当成一条完整 MAR 的耗时。

从项目根目录启动：

```bash
python3 deploy/qwen/start.py
curl http://127.0.0.1:19888/v1/models
python3 deploy/qwen/smoke.py
python3 deploy/qwen/benchmark_parallel.py
python3 deploy/qwen/benchmark_throughput.py --concurrency 8
```

后台启动器创建独立进程会话并把日志写入文件，关闭终端后模型服务会继续运行。

也可以在前台运行 `bash scripts/serve_qwen.sh`。该脚本允许用
`MAR_QWEN_GPU`、`MAR_QWEN_GPU_MEMORY`、`MAR_QWEN_CONTEXT`、`MAR_QWEN_MAX_SEQS`、
`MAR_QWEN_PORT` 覆盖相应配置；默认值与项目集成一致。

重新下载或核验：

```bash
python3 scripts/download_model.py --workers 6
python3 scripts/download_model.py --verify-only
```

下载器先读取固定 Hugging Face revision 的元数据。默认从官方
ModelScope 同名仓库下载大文件，使用其固定文件 revision，并要求每个文件的
SHA-256 和大小与 Hugging Face 元数据一致。小文件校验 Git blob SHA-1，
所有已完成文件另外记录 SHA-256。可用 `--source huggingface` 直接下载。
下载断点写在模型目录内，不额外复制一份权重到全局缓存。

新机器安装环境：

```bash
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install --no-cache-dir -r deploy/qwen/requirements.lock.txt
```

当前方案使用 Python 3.12、vLLM 0.32.0、Transformers 5.19.0、
PyTorch 2.13.0+cu130。启动脚本把此虚拟环境内的 CUDA toolkit 和 ninja
加入自己的环境，原项目及其他服务环境不受修改。

参考：[官方模型](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)、
[vLLM 模型部署说明](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)、
[vLLM 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。
