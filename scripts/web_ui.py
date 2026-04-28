from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.inference import default_device, default_weights_path, run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a local UI for tool detection.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def predict_from_ui(
    image_rgb: np.ndarray | None,
    weights_path: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    soft_nms: bool,
    retinex: bool,
):
    if image_rgb is None:
        raise gr.Error("请先上传一张图片。")

    image_bgr = image_rgb[:, :, ::-1].copy()
    result = run_inference(
        image_bgr=image_bgr,
        weights=Path(weights_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device.strip() or default_device(),
        soft_nms=soft_nms,
        retinex=retinex,
    )

    table_rows = []
    for item in result["detections"]:
        table_rows.append(
            [
                item["class_id"],
                item["class_name_zh"],
                item["class_name"],
                round(float(item["confidence"]), 4),
                str(item["bbox_xyxy"]),
            ]
        )

    rendered_rgb = result["image"][:, :, ::-1]
    return rendered_rgb, table_rows, result["summary"]


def build_demo() -> gr.Blocks:
    default_weights = str(default_weights_path())
    default_run_dir = str(Path(default_weights).parents[1])

    with gr.Blocks(title="YOLOv8 刀具识别演示") as demo:
        gr.Markdown(
            """
            # YOLOv8 刀具识别演示
            上传一张刀具图片，页面会调用你当前训练好的模型进行预测，并展示检测框和识别结果。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="上传待预测图片")
                weights_path = gr.Textbox(value=default_weights, label="模型权重路径")
                device = gr.Textbox(value=default_device(), label="设备")
                imgsz = gr.Slider(minimum=320, maximum=960, step=32, value=640, label="推理尺寸")
                conf = gr.Slider(minimum=0.05, maximum=0.9, step=0.05, value=0.25, label="置信度阈值")
                iou = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.5, label="IoU 阈值")
                soft_nms = gr.Checkbox(value=False, label="启用 Soft-NMS")
                retinex = gr.Checkbox(value=False, label="推理前做 Retinex")
                run_button = gr.Button("开始预测", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="numpy", label="预测结果")
                output_table = gr.Dataframe(
                    headers=["类别ID", "中文名称", "英文名称", "置信度", "边界框 xyxy"],
                    datatype=["number", "str", "str", "number", "str"],
                    row_count=(0, "dynamic"),
                    col_count=(5, "fixed"),
                    label="检测明细",
                )
                output_summary = gr.Textbox(label="结果摘要")

        gr.Markdown(
            f"当前默认成品模型目录：`{default_run_dir}`"
        )

        run_button.click(
            fn=predict_from_ui,
            inputs=[input_image, weights_path, device, imgsz, conf, iou, soft_nms, retinex],
            outputs=[output_image, output_table, output_summary],
        )

    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
