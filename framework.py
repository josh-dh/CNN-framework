import numpy as np

#convlayer

#input shape: (rows, columns, color: 3 if rgb, 1 if greyscale)

#initialize filters with gaussian noise
def init_filters(filter_count, filter_dimensions, rgb):
	if rgb: #rgb
		return np.array([np.random.normal(scale=50,size=filter_dimensions + (3,)) for i in range(filter_count)])
	else: #greyscale
		return np.array([np.random.normal(scale=50,size=filter_dimensions) for i in range(filter_count)])

def conv_layer(inputimage, filters, zero_pad_dimensions=(0,0), stride=(1,1), train=False, rgb=True):
	#zero pad equally on x and y axis equally per axis
	def zero_pad(inputimage, zero_pad_dimensions, rgb):
		if rgb: #rgb
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputimage.shape[1],inputimage.shape[2])),
					inputimage,
					np.zeros((zero_pad_dimensions[0],inputimage.shape[1], inputimage.shape[2]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2]))), axis=0)
		else: #greyscale
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputimage.shape[1])),
					inputimage,
					np.zeros((zero_pad_dimensions[0],inputimage.shape[1]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1]))), axis=0)
	#convolute filters across image and return result
	def convolute(inputimage, filters, stride):
		output = np.zeros((inputimage.shape[0]/stride[0],inputimage.shape[1]/stride[1], filters.shape[0]))
		for i in range(filters.shape[0]):
			for j in range(inputimage.shape[0]/stride[0]): #rows
				for k in range(inputimage.shape[1]/stride[1]): #columns
					output[j,k,i] = np.vdot(
						inputimage[np.ix_(np.arange(j*stride[0], j*stride[0] + filters.shape[1]), np.arange(k*stride[1], k*stride[1] + filters.shape[2]))],
						filters[i])
		return output

	return convolute(zero_pad(inputimage, zero_pad_dimensions, rgb), filters, stride)

conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), init_filters(5, (2,2), True), zero_pad_dimensions=(2,2))
