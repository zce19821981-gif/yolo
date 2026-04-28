## LabelImg 标注使用

项目标注相关文件：

- 类别文件：`configs/labelimg_classes.txt`
- 原始图片：`data`
- 标签输出：`labels`
- 启动脚本：`start_labelimg.sh` / `start_labelimg.bat`

### 启动方式

macOS / Linux:

```bash
bash start_labelimg.sh
```

只打开某一类，例如车刀：

```bash
bash start_labelimg.sh 车刀
```

打开混合场景目录，例如一张图里有多个刀具：

```bash
bash start_labelimg.sh data/mixed_scenes labels/mixed_scenes
```

Windows:

```bat
start_labelimg.bat
```

Windows 打开混合场景目录：

```bat
start_labelimg.bat data\mixed_scenes labels\mixed_scenes
```

### 标注建议

- 先从每类最清晰的图片开始
- 框只包住刀具本体，不要留太多背景
- 一张图有多个刀具，就画多个框
- 每个框单独选择类别，这样同一张图里可以同时标“车刀”“铣刀”“丝锥”等不同刀具
- 保存格式保持为 `YOLO`

### 混合场景目录建议

如果你的图片里会同时出现多个刀具，推荐直接单独建一个目录：

```text
data/
  mixed_scenes/
    scene_001.jpg
    scene_002.jpg
labels/
  mixed_scenes/
    scene_001.txt
    scene_002.txt
```

这样可以避免沿用旧的“按类别分文件夹”思路，比较适合检测任务。

### YOLO 标签示例

同一张图里每个目标写一行：

```text
0 0.245000 0.510000 0.180000 0.420000
11 0.742000 0.496000 0.205000 0.388000
5 0.531000 0.761000 0.120000 0.164000
```

上面表示同一张图片里有 3 个刀具框，类别分别是：

- `0` = `turning_tool`
- `11` = `end_mill`
- `5` = `tap`

### 后续训练

混合场景标注完成后，可以直接整理成训练集：

```bash
python scripts/prepare_dataset.py \
  --source-images data/mixed_scenes \
  --source-labels labels/mixed_scenes \
  --output datasets/tool15_user
```

这类数据不需要再走 `scripts/import_user_data.py`，因为那个脚本更适合“单类文件夹 + 单刀具图片”的旧流程。

训练时使用：

```bash
python scripts/train.py \
  --data configs/data/tool15_user.yaml \
  --model yolov8s.pt
```
