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


img1= Image.open('/root/autodl-tmp/DustGAN/datasets/original/ori_1.jpg')
img2 = Image.open('/root/autodl-tmp/DustGAN/test_img/ori_1_0.jpg')
img1 = img1.resize((480,640))
img2 = img2.resize((480,640))
i1_array = numpy.array(img1)
i2_array = numpy.array(img2)
r12 = psnr(i1_array, i2_array)
img1 = np.array(img1)
img2 = np.array(img2)
s12 = calculate_ssim(img1, img2)

img3= Image.open('/root/autodl-tmp/DustGAN/datasets/original/ori_3.jpg')
img4 = Image.open('/root/autodl-tmp/DustGAN/test_img/ori_3_0.jpg')
img3 = img3.resize((480,640))
img4 = img4.resize((480,640))
i3_array = numpy.array(img3)
i4_array = numpy.array(img4)
r34 = psnr(i3_array, i4_array)
img3 = np.array(img3)
img4 = np.array(img4)
s34 = calculate_ssim(img3, img4)

img5= Image.open('/root/autodl-tmp/DustGAN/datasets/original/ori_5.jpg')
img6 = Image.open('/root/autodl-tmp/DustGAN/test_img/ori_5_0.jpg')
img5 = img5.resize((480,640))
img6 = img6.resize((480,640))
i5_array = numpy.array(img5)
i6_array = numpy.array(img6)
r56 = psnr(i5_array, i6_array)
img5 = np.array(img5)
img6 = np.array(img6)
s56 = calculate_ssim(img5, img6)



print("CBAM-PSNR-1:", r12)
print("CBAM-SSIM-1:", s12)
print("CBAM-PSNR-3:", r34)
print("CBAM-SSIM-3:", s34)
print("CBAM-PSNR-5:", r56)
print("CBAM-SSIM-5:", s56)
