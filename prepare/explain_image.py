import numpy as np
from PIL import Image

im = np.array(Image.open("Lenna.bmp"))
print(im.shape)

img = im.copy()
im_R = im.copy()
im_R[:, :, (1, 2)] = 0
im_G = im.copy()
im_G[:, :, (0, 2)] = 0
im_B = im.copy()
im_B[:, :, (0, 1)] = 0

# 横に並べて結合（どれでもよい）
im_RGB = np.concatenate((img, im_R, im_G, im_B), axis=1)
# im_RGB = np.hstack((im_R, im_G, im_B))
# im_RGB = np.c_['1', im_R, im_G, im_B]

pil_img_RGB = Image.fromarray(im_RGB)
pil_img_RGB.save("lena_split.png")
