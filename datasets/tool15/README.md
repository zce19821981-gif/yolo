# 数据集目录说明

`prepare_dataset.py` 执行后，这里会生成标准 YOLO 目录：

```text
tool15
├── images
│   ├── train
│   ├── val
│   └── test
├── labels
│   ├── train
│   ├── val
│   └── test
└── reports
```

标签采用 YOLO 标准格式：

```text
class_id x_center y_center width height
```

五个数都应为归一化结果，范围在 `0 ~ 1` 之间。
