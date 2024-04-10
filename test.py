import cv2
import numpy as np
from PIL import Image
import numpy
import math


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


img1 = Image.open('/root/autodl-tmp/DustGAN/datasets/original/ori_2.jpg')
img2 = Image.open('/root/autodl-tmp/DustGAN/test_img/ori_2_0.jpg')
img3 = Image.open('/root/autodl-tmp/DustGAN/result-original/ori_2.png')
img1 = img1.resize((480,640))
img2 = img2.resize((480,640))

w, h = img3.size
img3 = img3.crop((w//2, 0, w, h))
img3 = img3.resize((480,640))

img4 = Image.open('/root/autodl-tmp/DustGAN/result-g-l/ori_2.png')
w1, h1 = img4.size
img4 = img4.crop((w//2, 0, w, h))
img4 = img4.resize((480,640))

i1_array = numpy.array(img1)
i2_array = numpy.array(img2)
i3_array = numpy.array(img3)
i4_array = numpy.array(img4)

r12 = psnr(i1_array, i2_array)
r13 = psnr(i1_array, i3_array)
r14 = psnr(i1_array, i4_array)

img1 = np.array(img1)
img2 = np.array(img2)
img3 = np.array(img3)
img4 = np.array(img4)


s12 = calculate_ssim(img1, img2)
s13 = calculate_ssim(img1, img3)
s14 = calculate_ssim(img1, img4)

print("O-PSNR:", r13)
print("G-L-PSNR", r14)
print("CBAM-PSNR:", r12)
print("O-SSIM:", s13)
print("G-L-SSIM", s14)
print("CBAM-SSIM:", s12)
