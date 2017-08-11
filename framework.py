import numpy as np

#convlayer

#input shape: (rows, columns, color: 3 if rgb, 1 if greyscale)

def conv_layer(inputimage, zero_pad_dimensions, filter_count, filter_dimensions, train=False, init=False):
	#initialize filters with gaussian noise
	def init_filters(filter_count, filter_dimensions):
		return np.array([np.random.normal(scale=50,size=filter_dimensions) for i in range(filter_count)])
	#zero pad equally on x and y axis equally per axis
	def zero_pad(inputimage, zero_pad_dimensions):
		if len(inputimage.shape) == 3:
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputimage.shape[1],inputimage.shape[2])),
					inputimage,
					np.zeros((zero_pad_dimensions[0],inputimage.shape[1], inputimage.shape[2]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2]))), axis=0)
		else:
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputimage.shape[1])),
					inputimage,
					np.zeros((zero_pad_dimensions[0],inputimage.shape[1]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1]))), axis=0)