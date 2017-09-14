"""
Joshua Donelly-Higgins
"""

#TODO: weight, bias gradients; test backprop

import numpy as np

#CONVLAYER

def init_weights(filter_count, filter_dimensions):
	"""
	initialize filters with gaussian noise
	"""
	return np.array([np.random.normal(scale=5,size=filter_dimensions) for i in range(filter_count)])


def init_biases(bias_count):
	"""
	initialize biases with gaussian noise
	"""
	return np.random.normal(scale=5,size=bias_count)


def conv_layer(inputobject, filters, biases, zero_pad_dimensions=(0,0), stride=(1,1), train=False):
	"""`
	convolutional layer that produces the convolution of an object with weights, biases and other parameters
	"""
	def zero_pad(inputobject, zero_pad_dimensions):
		"""
		zero pad equally on x and y axis equally per axis
		"""
		if len(inputobject.shape) > 2: #multidemsional
			print(np.zeros((zero_pad_dimensions[0],inputobject.shape[1], inputobject.shape[2])).shape)
			print(inputobject.shape)
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
		output = np.zeros(((inputobject.shape[0]-filters.shape[1])/stride[0],(inputobject.shape[1]-filters.shape[2])/stride[1], filters.shape[0]))
		for i in range(filters.shape[0]):
			for j in range((inputobject.shape[0]-filters.shape[1])/stride[0]): #rows
				for k in range((inputobject.shape[1]-filters.shape[2])/stride[1]): #columns
					output[j,k,i] = (biases[i] + np.vdot(
						inputobject[np.ix_(np.arange(j*stride[0], j*stride[0] + filters.shape[1]), np.arange(k*stride[1], k*stride[1] + filters.shape[2]))],
						filters[i]))
		return output


	return convolute(zero_pad(inputobject, zero_pad_dimensions), filters, biases, stride)

def fulcon_layer(inputobject, weights, biases):
	"""
	a simple fully-connected layer
	"""
	output = numpy.zeros((weights,))
	for i in output:
		output[i] = numpy.dot(inputobject.flatten(), weights[i]) + biases[i]
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


#BACKPROP

def final_layer_error(predictions, labels, weighted_input):
	"""
	calculate the error of the final layer
	"""
	def d_loss_quadratic_d_activations(predictions, labels):
		"""
		derivative of quadratic loss with respect to activations
		"""
		return np.subtract(predictions, labels)


	def d_softmax_activations_d_weighted_input(weighted_input):
		"""
		derivate of softmax activations with respect to weighted inputs UNSURE IF CORRECT
		"""
		output = np.zeros((weighted_input.size,))
		c = np.sum(np.exp(weighted_input)) #constant representing softmax denominator
		for i in range(weighted_input.size):
			ctemp = c - np.exp(weighted_input[i])
			output[i] = (ctemp * np.exp(weighted_input[i]))/np.power((ctemp * np.exp(weighted_input[i])),2)
		return output

	#hadamard product of two functions:
	return np.multiply(d_loss_quadratic_d_activations(predictions, labels), d_softmax_activations_d_weighted_input(weighted_input))


def lower_layer_error(currentweights,currenterror,lowerinput):
	"""
	calculate the error of the layer lower to the last calculated layer
	"""
	def relu_prime(lowerinput): #UNSURE IF CORRECT
		return np.minimum(lowerinput, 0)

	return np.multiply(np.multiply(np.transpose(currentweights),currenterror),relu_prime(lowerinput))


def d_loss_d_weight():
	"""
	return an array of loss derivatives with respect to their weights
	"""
	pass #TEMP


def d_loss_d_bias(error_for_layer):
	"""
	return an array of loss derivatives with respect to their biases THIS FUNCTION ONLY SERVES THE PURPOSE OF DOCUMENTATION
	"""
	return error_for_layer


def loss_quadratic(predictions, labels):
	"""
	quadratic loss function for one-hot labels and softmax predictions
	"""
	return np.sum(np.square(np.subtract(labels, predictions)))/(np.exp2(labels.size))


def loss_cross_entropy(predictions, labels):
	"""
	cross entropy loss designed to work with one-hot labels and softmax predictions UNTESTED; UNUSED
	"""
	output = 0
	for i in range(len(predictions)):
		output -= labels[i] * np.log(predictions[i])
	return output/i

#test functions
print(final_layer_error(softmax(np.array([5,2,5,1])), np.array([0,1,0,0]), np.array([5,2,5,1])))
print(conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), init_weights(5, (2,2,3)), init_biases(5), zero_pad_dimensions=(2,2)))
print(conv_layer(
	relu(
		conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), init_weights(5, (2,2,3)), init_biases(5), zero_pad_dimensions=(2,2))),
	init_weights(32, (2,2,5)),
	init_biases(32),
	zero_pad_dimensions=(1,1)
	).shape)