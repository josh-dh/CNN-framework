#Layers Module

import numpy as np

def init_weights(filter_count, filter_dimensions):
	"""
	initialize filters with gaussian noise
	"""
	return np.array([np.random.normal(scale=0.01,size=filter_dimensions) for i in range(filter_count)])


def init_biases(bias_count):
	"""
	initialize biases with gaussian noise
	"""
	return np.random.normal(scale=0.01,size=bias_count)


def conv_layer(inputobject, filters, biases, zero_pad_dimensions=(0,0), stride=(1,1), train=False):
	"""`
	convolutional layer that produces the convolution of an object with weights, biases and other parameters
	"""
	def zero_pad(inputobject, zero_pad_dimensions):
		"""
		zero pad equally on x and y axis equally per axis
		"""
		if len(inputobject.shape) > 2: #multidemsional
			inner = np.concatenate( #concatenate along columns
					(np.zeros((inputobject.shape[0],zero_pad_dimensions[0],inputobject.shape[2])),
					inputobject,
					np.zeros((inputobject.shape[0],zero_pad_dimensions[0],inputobject.shape[2]))), axis=1)
			return np.concatenate( #concatenate along rows
				(np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1], inner.shape[2]))), axis=0)
		else: #uni-dimensional
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputobject.shape[1])),
					inputobject,
					np.zeros((zero_pad_dimensions[0],inputobject.shape[1]))), axis=1)
			return np.concatenate(
				(np.zeros((zero_pad_dimensions[1],inner.shape[1])),
				inner,
				np.zeros((zero_pad_dimensions[1],inner.shape[1]))), axis=0)


	def convolute(inputobject, filters, biases, stride):
		"""
		convolute filters across image and return result
		"""
		output = np.zeros(((inputobject.shape[0]-filters.shape[1])//stride[0],(inputobject.shape[1]-filters.shape[2])//stride[1], filters.shape[0]))
		for i in range(filters.shape[0]):
			for j in range((inputobject.shape[0]-filters.shape[1])//stride[0]): #rows
				for k in range((inputobject.shape[1]-filters.shape[2])//stride[1]): #columns
					output[j,k,i] = (biases[i] + np.vdot(
						inputobject[np.ix_(np.arange(j*stride[0], j*stride[0] + filters.shape[1]), np.arange(k*stride[1], k*stride[1] + filters.shape[2]))],
						filters[i]))
		return output


	return convolute(zero_pad(inputobject, zero_pad_dimensions), filters, biases, stride)

def fulcon_layer(inputobject, weights, biases):
	"""
	a simple fully-connected layer
	"""
	output = np.empty(weights.shape[0])
	for i in range(output.shape[0]):
		output[i] = np.dot(inputobject.flatten(), weights[i].flatten()) + biases[i]

	return output


def softmax(array):
	"""
	returns the softmax of a 1d array
	"""
	return np.divide(np.exp(array), np.sum(np.exp(array)))


def relu(array):
	"""
	relu activation function
	"""
	array[array < 0] = 0
	return array
