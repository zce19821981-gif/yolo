from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "tool15"

TOOL_CLASSES = [
    "turning_tool",
    "milling_cutter",
    "drill_bit",
    "reamer",
    "boring_tool",
    "tap",
    "broach",
    "hob",
    "slotting_tool",
    "planer_tool",
    "twist_drill",
    "end_mill",
    "alloy_saw_blade",
    "countersink",
    "counterbore_drill",
]

TOOL_CLASSES_ZH = {
    "turning_tool": "车刀",
    "milling_cutter": "铣刀",
    "drill_bit": "钻头",
    "reamer": "铰刀",
    "boring_tool": "镗刀",
    "tap": "丝锥",
    "broach": "拉刀",
    "hob": "滚刀",
    "slotting_tool": "插刀",
    "planer_tool": "刨刀",
    "twist_drill": "麻花钻",
    "end_mill": "立铣刀",
    "alloy_saw_blade": "合金锯片",
    "countersink": "锪钻",
    "counterbore_drill": "扩孔钻",
}

NUM_CLASSES = len(TOOL_CLASSES)

