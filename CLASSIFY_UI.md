## 分类预测界面

这是基于 `data/类别名/图片` 训练出来的分类模型界面。

项目里现在有两种分类界面：

- 桌面版：`scripts/classify_ui.py`
- Web 版：`scripts/classify_web_ui.py`

默认模型权重：

- `models/tool15_cls_user_v1_best.pt`

### 启动方式

桌面版：

macOS / Linux:

```bash
bash start_classify_ui.sh
```
zsh /Users/mac/html/yolo/start_classify_ui.sh


Windows:

```bat
start_classify_ui.bat
```

Web 版：

macOS / Linux:

```bash
bash start_classify_web_ui.sh
```

Windows:

```bat
start_classify_web_ui.bat
```

如果你要长期公开访问，推荐在 Windows 上使用 Web 版，并配合 Cloudflare Tunnel。

Windows 示例：

```bat
set YOLO_UI_USERNAME=demo
set YOLO_UI_PASSWORD=change-this-password
start_classify_web_ui.bat --host 127.0.0.1 --port 7861
```

然后把 Cloudflare Tunnel 指向：

```text
http://127.0.0.1:7861
```

### 使用方法

1. 点击“选择图片”
2. 点击“开始预测”
3. 查看顶部预测类别和下方 Top-5 结果
4. 如有需要，点击“保存结果图”

说明：

- 桌面版适合在本机直接操作
- Web 版适合通过浏览器远程访问
- Web 版支持通过 `YOLO_UI_USERNAME` 和 `YOLO_UI_PASSWORD` 开启登录
- 这是分类界面，不会画检测框
- 图片上方的覆盖文字现在使用英文，避免 OpenCV 中文乱码
