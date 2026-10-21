import os
from PyPDF2 import PdfReader, PdfWriter

def crop_top_left():
    pdf_path = os.path.join("results", "new_v13_ms_multiseed_plots", "new_v13_ms_multiseed_training_curves.pdf")
    out_path = os.path.join("results", "new_v13_ms_multiseed_plots", "new_v13_ms_multiseed_training_curves_cropped.pdf")

    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    page = reader.pages[0]
    
    # 获取原始页面的宽和高
    box = page.mediabox
    width = float(box.right) - float(box.left)
    height = float(box.top) - float(box.bottom)

    # 截取左上角 (2x2网格的第一个图)
    # y 轴向上为正，左上角的范围是：
    # x: 0 -> width/2
    # y: height/2 -> height
    page.mediabox.left = float(box.left)
    page.mediabox.bottom = float(box.bottom) + height / 2
    page.mediabox.right = float(box.left) + width / 2
    page.mediabox.top = float(box.top)

    writer.add_page(page)

    with open(out_path, "wb") as f:
        writer.write(f)

    print(f"Cropped PDF saved to {out_path}")

if __name__ == "__main__":
    crop_top_left()
