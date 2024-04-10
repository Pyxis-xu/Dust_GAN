import os
from PIL import Image

# 图像文件夹路径
image_dir = '/root/autodl-tmp/DustGAN/datasets/dust'

# 获取文件夹中所有图像列表
image_list = os.listdir(image_dir)

# 遍历每一个图像
for img_name in image_list:

    # 拼接图像路径
    img_path = os.path.join(image_dir, img_name)

    # 打开图像并转换为 RGB 彩色模式
    with Image.open(img_path) as img:

        # 如果是 4 通道图像（例如 transparent PNG），则进行转换
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # 覆盖保存图像
        img.save(img_path)
