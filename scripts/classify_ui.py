from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from tool_yolo.classify_inference import default_classify_weights_path, run_classification
from tool_yolo.inference import default_device


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Launch a desktop UI for tool classification.')
    parser.add_argument('--weights', type=Path, default=default_classify_weights_path())
    return parser.parse_args()


class ToolClassifierApp:
    def __init__(self, root: tk.Tk, default_weights: Path) -> None:
        self.root = root
        self.root.title('YOLOv8 刀具分类预测')
        self.root.geometry('1320x820')

        self.image_path_var = tk.StringVar()
        self.weights_path_var = tk.StringVar(value=str(default_weights))
        self.device_var = tk.StringVar(value=default_device())
        self.imgsz_var = tk.StringVar(value='224')
        self.summary_var = tk.StringVar(value='请选择一张图片开始预测。')

        self.original_photo: ImageTk.PhotoImage | None = None
        self.result_photo: ImageTk.PhotoImage | None = None
        self.result_image_bgr = None

        self._build_layout()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        control = ttk.LabelFrame(main, text='参数与操作', padding=12)
        control.pack(fill=tk.X)

        ttk.Label(control, text='图片路径').grid(row=0, column=0, sticky='w')
        ttk.Entry(control, textvariable=self.image_path_var, width=84).grid(row=0, column=1, columnspan=4, sticky='ew', padx=6)
        ttk.Button(control, text='选择图片', command=self.select_image).grid(row=0, column=5, padx=6)

        ttk.Label(control, text='权重路径').grid(row=1, column=0, sticky='w')
        ttk.Entry(control, textvariable=self.weights_path_var, width=84).grid(row=1, column=1, columnspan=4, sticky='ew', padx=6)
        ttk.Button(control, text='选择权重', command=self.select_weights).grid(row=1, column=5, padx=6)

        ttk.Label(control, text='设备').grid(row=2, column=0, sticky='w')
        ttk.Entry(control, textvariable=self.device_var, width=12).grid(row=2, column=1, sticky='w', padx=6)
        ttk.Label(control, text='尺寸').grid(row=2, column=2, sticky='w')
        ttk.Entry(control, textvariable=self.imgsz_var, width=12).grid(row=2, column=3, sticky='w', padx=6)
        ttk.Button(control, text='开始预测', command=self.run_prediction).grid(row=2, column=4, padx=6)
        ttk.Button(control, text='保存结果图', command=self.save_result).grid(row=2, column=5, padx=6)

        preview = ttk.Frame(main)
        preview.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        original_frame = ttk.LabelFrame(preview, text='原图', padding=8)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.original_label = ttk.Label(original_frame, anchor='center')
        self.original_label.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(preview, text='预测结果', padding=8)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))
        self.result_label = ttk.Label(result_frame, anchor='center')
        self.result_label.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        summary_frame = ttk.LabelFrame(bottom, text='结果摘要', padding=8)
        summary_frame.pack(fill=tk.X)
        ttk.Label(summary_frame, textvariable=self.summary_var, wraplength=1240).pack(anchor='w')

        table_frame = ttk.LabelFrame(bottom, text='Top-5 预测', padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        columns = ('rank', 'class_id', 'class_name_zh', 'class_name', 'confidence')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        self.tree.heading('rank', text='排名')
        self.tree.heading('class_id', text='类别ID')
        self.tree.heading('class_name_zh', text='中文名称')
        self.tree.heading('class_name', text='英文名称')
        self.tree.heading('confidence', text='置信度')
        self.tree.column('rank', width=80, anchor='center')
        self.tree.column('class_id', width=80, anchor='center')
        self.tree.column('class_name_zh', width=160, anchor='center')
        self.tree.column('class_name', width=220, anchor='center')
        self.tree.column('confidence', width=120, anchor='center')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        for column in range(6):
            control.columnconfigure(column, weight=1)

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title='选择待预测图片',
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.webp')],
        )
        if file_path:
            self.image_path_var.set(file_path)
            image = cv2.imread(file_path)
            if image is not None:
                self._set_preview(self.original_label, image, is_result=False)

    def select_weights(self) -> None:
        file_path = filedialog.askopenfilename(
            title='选择分类模型权重',
            filetypes=[('PyTorch weights', '*.pt')],
        )
        if file_path:
            self.weights_path_var.set(file_path)

    def _set_preview(self, label: ttk.Label, image_bgr, *, is_result: bool) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((600, 420))
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        if is_result:
            self.result_photo = photo
        else:
            self.original_photo = photo

    def run_prediction(self) -> None:
        image_path = Path(self.image_path_var.get().strip())
        weights_path = Path(self.weights_path_var.get().strip())

        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            messagebox.showerror('错误', '请先选择一张有效图片。')
            return

        try:
            imgsz = int(self.imgsz_var.get().strip())
        except ValueError:
            messagebox.showerror('错误', '尺寸必须是合法整数。')
            return

        image = cv2.imread(str(image_path))
        if image is None:
            messagebox.showerror('错误', f'无法读取图片: {image_path}')
            return

        try:
            result = run_classification(
                image_bgr=image,
                weights=weights_path,
                imgsz=imgsz,
                device=self.device_var.get().strip() or default_device(),
            )
        except Exception as exc:
            messagebox.showerror('预测失败', str(exc))
            return

        self._set_preview(self.original_label, image, is_result=False)
        self._set_preview(self.result_label, result['image'], is_result=True)
        self.result_image_bgr = result['image']
        self.summary_var.set(str(result['summary']))

        for item in self.tree.get_children():
            self.tree.delete(item)

        for rank, item in enumerate(result['topk'], start=1):
            self.tree.insert(
                '',
                tk.END,
                values=(
                    rank,
                    item['class_id'],
                    item['class_name_zh'],
                    item['class_name'],
                    f"{item['confidence']:.4f}",
                ),
            )

    def save_result(self) -> None:
        if self.result_image_bgr is None:
            messagebox.showinfo('提示', '请先完成一次预测。')
            return

        save_path = filedialog.asksaveasfilename(
            title='保存预测结果',
            defaultextension='.jpg',
            filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png')],
        )
        if not save_path:
            return

        cv2.imwrite(save_path, self.result_image_bgr)
        messagebox.showinfo('成功', f'预测结果已保存到\n{save_path}')


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    width, height = 1320, 820
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = max((screen_w - width) // 2, 0)
    y = max((screen_h - height) // 2, 0)
    root.geometry(f'{width}x{height}+{x}+{y}')
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.focus_force()
    ToolClassifierApp(root=root, default_weights=args.weights)
    root.after(800, lambda: root.attributes('-topmost', False))
    root.mainloop()


if __name__ == '__main__':
    main()
