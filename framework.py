import numpy as np

#convlayer

#input shape: (rows, columns, color: 3 if rgb, 1 if greyscale)

#initialize filters with gaussian noise
def init_filters(filter_count, filter_dimensions):
	return np.array([np.random.normal(scale=5,size=filter_dimensions) for i in range(filter_count)])

#initialize
def init_biases(bias_count):
	return np.random.normal(scale=100,size=bias_count)

#convolutional layer
def conv_layer(inputimage, filters, biases, zero_pad_dimensions=(0,0), stride=(1,1), train=False):
	#zero pad equally on x and y axis equally per axis
	def zero_pad(inputimage, zero_pad_dimensions):
		if len(inputimage.shape) > 2: #multidemsional
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputimage.shape[1],inputimage.shape[2])),
					inputimage,
					np.zeros((zero_pad_dimensions[0],inputimage.shape[1], inputimage.shape[2]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2]))), axis=0)
		else: #uni-dimensional
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputimage.shape[1])),
					inputimage,
					np.zeros((zero_pad_dimensions[0],inputimage.shape[1]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1]))), axis=0)
	#convolute filters across image and return result
	def convolute(inputimage, filters, biases, stride):
		output = np.zeros(((inputimage.shape[0]-filters.shape[1])/stride[0],(inputimage.shape[1]-filters.shape[2])/stride[1], filters.shape[0]))
		for i in range(filters.shape[0]):
			for j in range((inputimage.shape[0]-filters.shape[1])/stride[0]): #rows
				for k in range((inputimage.shape[1]-filters.shape[2])/stride[1]): #columns
					output[j,k,i] = (biases[i] + np.vdot(
						inputimage[np.ix_(np.arange(j*stride[0], j*stride[0] + filters.shape[1]), np.arange(k*stride[1], k*stride[1] + filters.shape[2]))],
						filters[i]))
		return output

	return convolute(zero_pad(inputimage, zero_pad_dimensions), filters, biases, stride)

#cross entropy loss designed to work with one-hot labels and softmax predictions UNTESTED
def loss(predictions, labels):
	output = 0
	for i in range(len(predictions)):
		output += labels[i] * 1/np.log(predictions[i])
	return output

#returns the softmax of a 1d array UNTESTED
def softmax(array):
	return np.divide(array, np.sum(exp(array)))

print(conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), init_filters(5, (2,2,3)), init_biases(5), zero_pad_dimensions=(2,2)))