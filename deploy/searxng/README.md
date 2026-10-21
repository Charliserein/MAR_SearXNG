# SearXNG 本机部署

服务地址为 `http://127.0.0.1:18888`，容器名为 `mar-cached-searxng`。这是本项目独立实例，未修改原 MAR 项目和其他容器。

在项目根目录执行：

```bash
./scripts/searxng.sh up
./scripts/searxng.sh status
./scripts/searxng.sh check
```

修改 `settings.yml` 后用 `./scripts/searxng.sh restart` 重载；用 `./scripts/searxng.sh down` 仅停止本项目的 SearXNG。停止不会删除持久卷或 MAR 的证据缓存。

`check` 会实际联网发出一次网页检索和一次图片检索，并将结果数量、成功引擎、失败引擎、示例链接、时间和镜像信息保存到 `outputs/searxng-check.json`。它不会填充 MAR 的 SQLite 证据缓存；缓存由项目检索客户端负责。

## 能力与限制

网页检索使用 `bing`、`brave`、`duckduckgo`；图片类别默认使用 `bing images`。`duckduckgo images` 实测可返回图片，但存在间歇超时，因此保留配置并默认关闭。首次探测中 Google 两个引擎返回 403，因此已移除，初始结果保存在 `outputs/searxng-initial-check.json`。

图片类别接收的是**文字查询**，返回相关图片和来源网页。官方当前没有可直接替代 Serper Lens 的上传图片或图片 URL 反向搜索接口。MAR 的真正反向搜图证据仍由作者已有图注和 Serper 缓存提供；SearXNG 可提供网页检索和视觉描述文本搜图。

```bash
curl --noproxy '*' --get 'http://127.0.0.1:18888/search' \
  --data-urlencode 'q=NASA Artemis I launch November 16 2022' \
  --data-urlencode 'categories=images' \
  --data-urlencode 'format=json'
```

## 本机网络与持久化

- 当前 Linux 主机的外网 HTTP 代理为 `http://127.0.0.1:17897`。容器采用 host 网络以访问已有代理，同时用 `GRANIAN_HOST=127.0.0.1` 严格限制服务监听地址。`settings.yml` 中的 `outgoing.proxies` 指向该代理。
- 若迁移到另一台机器，需要按目标环境修改或删除 `outgoing.proxies`。本部署依赖该代理保持运行。
- 调用本机接口应绕过全局代理：HTTPX 使用 `trust_env=False`，curl 使用 `--noproxy '*'`，或配置合适的 `NO_PROXY`。检查脚本已使用无代理的 urllib opener。
- `search.formats` 开启 `html` 与 `json`；本机私有实例关闭 limiter。实例没有向公网开放。
- 实例密钥在首次启动时随机生成于 `instance.env`，权限为 `600`，此文件已单独加入 `.gitignore`，不要提交或复制其内容到日志。
- 官方 SearXNG 自身数据保存在 Docker 卷 `mar-cached-search_searxng-data`。这与 MAR 按查询/图片哈希存储的证据缓存相互独立。

## 固定版本与依据

部署时官方版本：`2026.10.19+81371861f`。Compose 固定 Docker Hub 镜像摘要：

```text
docker.io/searxng/searxng@sha256:5ed0323b5f5f8701264ef65e239a2bd83cb57098f2b971fa30bf79c06eeae271
```

- [官方容器安装文档](https://docs.searxng.org/admin/installation-docker.html)
- [官方 Search API 与 JSON 配置](https://docs.searxng.org/dev/search_api.html)
- [官方尚未关闭的反向搜图需求](https://github.com/searxng/searxng/issues/291)
- [Google Images 引擎：将文字 query 发送到图片搜索](https://github.com/searxng/searxng/blob/master/searx/engines/google_images.py)
