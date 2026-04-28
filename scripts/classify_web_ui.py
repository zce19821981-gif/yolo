from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from tool_yolo.classify_inference import default_classify_weights_path, run_classification
from tool_yolo.inference import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Launch a local web UI for tool classification.')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--auth-user', type=str, default='')
    parser.add_argument('--auth-pass', type=str, default='')
    return parser.parse_args()


def resolve_auth(args: argparse.Namespace) -> tuple[str, str] | None:
    username = args.auth_user.strip() or os.getenv('YOLO_UI_USERNAME', '').strip()
    password = args.auth_pass.strip() or os.getenv('YOLO_UI_PASSWORD', '').strip()
    if not username and not password:
        return None
    if not username or not password:
        raise ValueError('Both username and password are required for web UI auth.')
    return username, password


def predict_from_ui(image_rgb: np.ndarray | None, weights_path: str, device: str, imgsz: int):
    if image_rgb is None:
        raise gr.Error('请先上传一张图片。')

    image_bgr = image_rgb[:, :, ::-1].copy()
    result = run_classification(
        image_bgr=image_bgr,
        weights=Path(weights_path),
        imgsz=imgsz,
        device=device.strip() or default_device(),
    )

    rows = []
    for rank, item in enumerate(result['topk'], start=1):
        rows.append([
            rank,
            item['class_id'],
            item['class_name_zh'],
            item['class_name'],
            round(float(item['confidence']), 4),
        ])

    rendered_rgb = result['image'][:, :, ::-1]
    return rendered_rgb, rows, result['summary']


def build_demo() -> gr.Blocks:
    default_weights = str(default_classify_weights_path())

    with gr.Blocks(title='YOLOv8 刀具分类演示') as demo:
        gr.Markdown(
            '''
            # YOLOv8 刀具分类演示
            上传一张刀具图片，页面会调用当前训练好的分类模型进行预测，并展示 Top-5 结果。
            '''
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type='numpy', label='上传待预测图片')
                weights_path = gr.Textbox(value=default_weights, label='模型权重路径')
                device = gr.Textbox(value=default_device(), label='设备')
                imgsz = gr.Slider(minimum=128, maximum=512, step=32, value=224, label='推理尺寸')
                run_button = gr.Button('开始预测', variant='primary')

            with gr.Column(scale=1):
                output_image = gr.Image(type='numpy', label='预测结果')
                output_table = gr.Dataframe(
                    headers=['排名', '类别ID', '中文名称', '英文名称', '置信度'],
                    datatype=['number', 'number', 'str', 'str', 'number'],
                    row_count=(0, 'dynamic'),
                    col_count=(5, 'fixed'),
                    label='Top-5 预测',
                )
                output_summary = gr.Textbox(label='结果摘要')

        run_button.click(
            fn=predict_from_ui,
            inputs=[input_image, weights_path, device, imgsz],
            outputs=[output_image, output_table, output_summary],
        )

    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo()
    auth = resolve_auth(args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth)


if __name__ == '__main__':
    main()
