## 长期公开访问分类 Web UI

推荐方案：

- Windows 上运行 `scripts/classify_web_ui.py`
- Cloudflare Tunnel 只暴露 `http://127.0.0.1:7861`
- Cloudflare Access 作为第一层登录
- Gradio `auth` 作为第二层登录

### 1. 在 Windows 上启动 Web UI

先安装依赖：

```bat
pip install -r requirements.txt
pip install -r requirements-webui.txt
```

再设置登录账号并启动：

```bat
set YOLO_UI_USERNAME=demo
set YOLO_UI_PASSWORD=change-this-password
start_classify_web_ui.bat --host 127.0.0.1 --port 7861
```

说明：

- 这里监听 `127.0.0.1`，只允许本机访问
- 对外公开交给 Cloudflare Tunnel，不直接暴露 Windows 端口

### 2. 安装 Cloudflare Tunnel

安装 `cloudflared` 后登录：

```bat
cloudflared tunnel login
```

创建 tunnel：

```bat
cloudflared tunnel create yolo-classify-ui
```

### 3. 配置 tunnel

在 Windows 用户目录下创建配置文件：

```text
%USERPROFILE%\.cloudflared\config.yml
```

内容示例：

```yaml
tunnel: yolo-classify-ui
credentials-file: C:\Users\Administrator\.cloudflared\<your-tunnel-id>.json

ingress:
  - hostname: yolo.your-domain.com
    service: http://127.0.0.1:7861
  - service: http_status:404
```

然后把 DNS 绑定到 tunnel：

```bat
cloudflared tunnel route dns yolo-classify-ui yolo.your-domain.com
```

启动 tunnel：

```bat
cloudflared tunnel run yolo-classify-ui
```

### 4. 配置 Cloudflare Access

在 Cloudflare Zero Trust 里给 `yolo.your-domain.com` 创建一个 Self-hosted application。

建议：

- 至少要求邮箱登录
- 只允许你指定的邮箱域名或指定邮箱账号
- 不要给这个域名配置任何 SSH 或 RDP 相关入口

### 5. 访问方式

外部用户访问：

```text
https://yolo.your-domain.com
```

访问流程：

1. 先通过 Cloudflare Access 登录
2. 再输入 Gradio 用户名和密码
3. 进入 YOLO 分类 Web UI

### 安全建议

- 不要把 UI 绑定到 `0.0.0.0` 再直接做公网端口映射
- 不要把 `22`、`3389`、文件共享端口暴露到公网
- 生产环境优先使用单独机器或单独 Windows 账户运行这套服务
