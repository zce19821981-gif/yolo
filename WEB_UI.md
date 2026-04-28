## Web UI

这是项目里的网页预测界面脚本入口：

- `scripts/web_ui.py`

### 本地启动

macOS / Linux / Windows 都可以直接用 Python 运行：

```bash
python scripts/web_ui.py
```

如果你要启用分享链接：

```bash
python scripts/web_ui.py --share
```

### 依赖

```bash
pip install -r requirements-webui.txt
```

### 建议

如果后续要让老师或同学跨平台直接访问，最推荐把这个界面部署到 Hugging Face Spaces 或其他公开 Web 平台。
