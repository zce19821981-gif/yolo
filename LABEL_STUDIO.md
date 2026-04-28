## Label Studio 标注

推荐在你这台机器上使用 Label Studio 做网页标注，因为它比当前的 LabelImg 桌面环境稳定。

### 启动

```bash
cd /Users/mac/html/yolo
bash start_labelstudio.sh
```

如果 `8080` 已经被占用，可以换端口：

```bash
bash start_labelstudio.sh 9001
```

启动后打开：

```text
http://127.0.0.1:8080
```

### 首次创建项目

1. 创建管理员账号并登录
2. 新建项目
3. 在 `Labeling Setup` 里粘贴这个文件内容：

- `configs/label_studio_tool15.xml`

这个配置已经按你当前 YOLO 的 15 个类别预设好了，并且固定了导出顺序，导出 YOLO 时会尽量对齐现有类别 id。

### 导入本地图片

官方本地存储需要：

- `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`
- `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` 指向图片目录的父目录

启动脚本已经替你设置好了，根目录就是：

```text
/Users/mac/html/yolo
```

如果你的图片放在：

```text
/Users/mac/html/yolo/data/mixed_scenes
```

那么在项目里添加本地源存储时：

- 选择 `Local Files`
- `Absolute local path` 填 `/Users/mac/html/yolo/data/mixed_scenes`
- `Import Method` 选 `Files`
- `File Name Filter` 可填 `.*`
- 如果有子目录，勾选 `Scan all sub-folders`

### 标注方式

- 一张图里有几个刀具，就画几个框
- 每个框单独选择对应类别
- 推荐使用 `RectangleLabels`

### 导出 YOLO

标注完成后在项目里导出：

- `YOLO`

如果你还想把原图一起打包导出，也可以在导出界面选择带图片的选项。

### 和当前训练流程对接

导出后，把 `images/` 和 `labels/` 放到你想整理的原始目录下，再执行：

```bash
python scripts/prepare_dataset.py \
  --source-images <exported-images-dir> \
  --source-labels <exported-labels-dir> \
  --output datasets/tool15_user
```
