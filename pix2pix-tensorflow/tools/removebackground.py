from skimage import io as skio
url = 'data/resized/f-021-01.png'
img = skio.imread(url)

print("shape of image: {}".format(img.shape))
print("dtype of image: {}".format(img.dtype))

#detecting edges
from skimage import filters
sobel = filters.sobel(img)
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200
blurred = filters.gaussian(sobel, sigma=2.0)

#obtaining seeds
import numpy as np
light_spots = np.array((img > 245).nonzero()).T
dark_spots = np.array((img < 3).nonzero()).T

#making a seed mask
from scipy import ndimage as ndi
bool_mask = np.zeros(img.shape, dtype=np.bool)
bool_mask[tuple(light_spots.T)] = True
bool_mask[tuple(dark_spots.T)] = True
seed_mask, num_seeds = ndi.label(bool_mask)
num_seeds

#applying watershed
from skimage import morphology
ws = morphology.watershed(blurred, seed_mask)

background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
background_mask = (ws == background)
cleaned = img * ~background_mask

fig.savefig('data/plot.png')


