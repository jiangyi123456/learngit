# encoding: utf-8
# import string
# import re
# import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=np.array(Image.open('/home/jy/programepy/picture1/65_mip.tif'))
plt.figure("beauty")
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()