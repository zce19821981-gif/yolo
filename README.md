# YOLOv8 刀具识别项目

这个仓库同时保留了两条路线：

- 检测路线：基于 YOLOv8 的刀具检测训练、验证、预测
- 分类路线：基于 `data/类别名/图片` 的刀具分类训练与桌面预测

当前最稳定、最完整的成品是分类路线。你已经可以直接用训练好的分类模型进行本地预测，也可以把项目上传到 GitHub 后给其他用户在 macOS 或 Windows 上复现。

## 项目结构

```text
yolo
├── configs
│   ├── data
│   └── hyp
├── data
├── datasets
├── models
├── runs
├── scripts
├── src
├── start_classify_ui.sh
├── start_classify_ui.bat
├── start_classify_web_ui.sh
├── start_classify_web_ui.bat
├── start_desktop_ui.sh
├── start_desktop_ui.bat
├── start_labelimg.sh
└── start_labelimg.bat
```

## 环境安装

先进入项目根目录：

```bash
cd yolo
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果你要运行网页界面，再额外安装：

```bash
pip install -r requirements-webui.txt
```

## 当前分类成品

当前默认分类权重已经固定放在：

- `models/tool15_cls_user_v1_best.pt`

这样即使不上传 `runs/`，项目也能直接加载成品模型。

## 直接运行分类预测界面

桌面版：

### macOS / Linux

```bash
bash start_classify_ui.sh
```

### Windows

```bat
start_classify_ui.bat
```

界面脚本本体在：

- `scripts/classify_ui.py`

Web 版：

### macOS / Linux

```bash
bash start_classify_web_ui.sh
```

### Windows

```bat
start_classify_web_ui.bat
```

网页界面脚本本体在：

- `scripts/classify_web_ui.py`

如果你要把分类界面长期公开给外部用户访问，推荐在 Windows 上这样启动：

```bat
set YOLO_UI_USERNAME=demo
set YOLO_UI_PASSWORD=change-this-password
start_classify_web_ui.bat --host 127.0.0.1 --port 7861
```

然后再把 Cloudflare Tunnel 指向 `http://127.0.0.1:7861`。

## 直接运行检测预测界面

### macOS / Linux

```bash
bash start_desktop_ui.sh
```

### Windows

```bat
start_desktop_ui.bat
```

说明：检测路线依赖检测权重和检测数据配置；如果你只做当前这版分类成品，优先使用分类界面。

## LabelImg 标注

### macOS / Linux

```bash
bash start_labelimg.sh
```

只打开某一类，例如车刀：

```bash
bash start_labelimg.sh 车刀
```

如果你要标“同一张图多个刀具”的检测数据，推荐新建一个混合场景目录，例如：

```bash
bash start_labelimg.sh data/mixed_scenes labels/mixed_scenes
```

### Windows

```bat
start_labelimg.bat
```

这种情况下：

- 一张图里有几个刀具，就画几个框
- 每个框单独选择类别
- 同一个 `.txt` 标签文件里会有多行，对应多个刀具

示例：

```text
0 0.245000 0.510000 0.180000 0.420000
11 0.742000 0.496000 0.205000 0.388000
5 0.531000 0.761000 0.120000 0.164000
```

## Label Studio 网页标注

如果你本机的 `LabelImg` 会闪退，推荐改用本地网页标注：

- 启动脚本：`start_labelstudio.sh`
- 预设标签配置：`configs/label_studio_tool15.xml`
- 使用说明：`LABEL_STUDIO.md`

启动：

```bash
bash start_labelstudio.sh
```

默认地址：

```text
http://127.0.0.1:8080
```

## 分类训练

你的原始数据格式是：

```text
data/
  车刀/
  铣刀/
  丝锥/
  ...
```

先整理分类数据集：

```bash
python scripts/prepare_classification_dataset.py \
  --source-root data \
  --output-root datasets/tool15_cls_user \
  --overwrite
```

再开始训练：

```bash
python scripts/train_classify.py \
  --data datasets/tool15_cls_user \
  --model yolov8s-cls.pt \
  --epochs 60
```

设备会自动选择：

- Apple Silicon 优先 `mps`
- NVIDIA 显卡优先 `cuda:0`
- 都没有时使用 `cpu`

## 检测训练

检测数据配置已经改成了相对路径，上传 GitHub 后仍可用：

- `configs/data/tool15.yaml`
- `configs/data/tool15_real_bootstrap.yaml`
- `configs/data/tool15_user.yaml`

如果你是自己新标的混合场景图片，可以先整理：

```bash
python scripts/prepare_dataset.py \
  --source-images data/mixed_scenes \
  --source-labels labels/mixed_scenes \
  --output datasets/tool15_user
```

然后训练你自己的用户数据：

```bash
python scripts/train.py \
  --data configs/data/tool15_user.yaml \
  --model yolov8s.pt \
  --epochs 200 \
  --imgsz 960 \
  --batch 8
```

示例：

```bash
python scripts/train.py \
  --data configs/data/tool15.yaml \
  --model yolov8s.pt \
  --epochs 200 \
  --imgsz 960 \
  --batch 8
```

## 上传到 GitHub 的建议

1. 不要上传 `.venv`
2. 可以上传 `models/tool15_cls_user_v1_best.pt` 作为成品模型
3. 如果后续模型更大，建议改用 Git LFS
4. 如果不想上传原始数据集，可以保留 `scripts`、`src`、`configs`、`models` 和文档

## 跨平台说明

这个仓库已经完成了下面这些迁移改造：

- 训练配置改成相对路径，不再绑定 `/Users/mac/...`
- 启动脚本新增了 `bash` 和 `.bat` 两套入口
- 分类模型默认路径固定为 `models/`，更适合分享
- 分类训练默认设备改成自动选择

如果你下一步要继续做“别人直接浏览器访问”，最推荐的是把 `scripts/web_ui.py` 再整理成公开部署版本，例如部署到 Hugging Face Spaces。
