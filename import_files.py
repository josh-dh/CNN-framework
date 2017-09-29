#Import Module

import numpy as np
from PIL import Image

def import_batch(path, numlow, numhigh):
	"""
	loads a batch of images from a range of filenames within a path
	"""
	def process_image_file(path):
		return np.array(Image.open(path))


	output = np.empty((numhigh-numlow,32,32,3))
	for i in range(numlow,numhigh):
		output[(i-numlow)] = process_image_file("%s%d.png" % (path, i))

	return output