import torch.nn as nn
import numpy as np
import torch
from dataloader import train_dataset
from PIL import Image

import matplotlib.pyplot as plt

img = Image.open('fuji.jpg')
img = img.resize((64, 64))
img_src: np.ndarray = np.array(img)

img = np.transpose(img_src, axes=(2, 0, 1)).reshape(3, -1) / 255
print(img.shape)

s = np.dot(img.T, img)
print(s.shape)

# softmax
exp = np.exp(s)
sum = np.sum(exp, axis=1)
b = exp / (sum)


o = np.dot(img, b.T)

i = 300
sample = b[i].reshape(64, 64)
print(np.max(sample))
fig = plt.figure(dpi=200)
ax1: plt.Axes = fig.add_subplot(1, 2, 1)
ax1.set_title('Source Image')
ax1.imshow(img_src)
ax1.scatter([i % 64], [i//64], color='red')
ax1 = fig.add_subplot(1, 2, 2)
ax1.set_title(f'Attention Map for ({i%64}, {i//64})')
ax1.imshow(sample)
plt.show()
exit(0)


print(o.shape)

y = img + 0.5 * o
img2 = y.reshape(3, 64, 64).transpose(1, 2, 0)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img_src)
ax1 = fig.add_subplot(1, 2, 2)
ax1.imshow(img2)
plt.show()

# img = torch.tensor(img)
# print(img.shape)
# x = img.permute(2, 0, 1)
# x = img.view(x.shape[0], -1)

# s = torch.mm(x.T, x)
# softmax = nn.Softmax(dim=-2)
# b = softmax(s)
# plt.imshow(img)
# plt.show()
