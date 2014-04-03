__author__ = 'pedja'
__date__ = 'April 3rd 2014'

from skimage import data, io, filter
from scipy import ndimage
from skimage.morphology import watershed

import matplotlib.pyplot as plt
import os
import numpy as np

coins = data.coins()
# io.imshow(coins)
# io.show()
# io.imsave('coins.jpg', coins)
# edges = filter.sobel(image)
# print os.getcwd()
# io.imsave('edges.jpg', edges)
edges = filter.canny(coins/255.)
fill_coins = ndimage.binary_fill_holes(edges)
print type(edges)
print len(edges[edges == True])
print coins
# io.imshow(edges)
# io.show()
#
# io.imshow(fill_coins)
# io.show()


label_objects, nb_labels = ndimage.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
#
# io.imshow(coins_cleaned)
# io.show()

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2
elevation_map = filter.sobel(coins)
segmentation = watershed(elevation_map, markers)
segmentation = ndimage.binary_fill_holes(segmentation - 1)

io.imshow(segmentation)
io.show()

labeled_coins, _ = ndimage.label(segmentation)

io.imshow(labeled_coins)
io.show()