import numpy as np

#convlayer

#input shape: (rows, columns, color: 3 if rgb, 1 if greyscale)

#initialize filters with gaussian noise
def init_weights(filter_count, filter_dimensions):
	return np.array([np.random.normal(scale=5,size=filter_dimensions) for i in range(filter_count)])

#initialize
def init_biases(bias_count):
	return np.random.normal(scale=5,size=bias_count)

#convolutional layer
def conv_layer(inputobject, filters, biases, zero_pad_dimensions=(0,0), stride=(1,1), train=False):
	#zero pad equally on x and y axis equally per axis
	def zero_pad(inputobject, zero_pad_dimensions):
		if len(inputobject.shape) > 2: #multidemsional
			inner = np.concatenate(
					(np.zeros((zero_pad_dimensions[0],inputobject.shape[1],inputobject.shape[2])),
					inputobject,
					np.zeros((zero_pad_dimensions[0],inputobject.shape[1], inputobject.shape[2]))), axis=1)
			return np.concatenate(
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
	#convolute filters across image and return result
	def convolute(inputobject, filters, biases, stride):
		output = np.zeros(((inputobject.shape[0]-filters.shape[1])/stride[0],(inputobject.shape[1]-filters.shape[2])/stride[1], filters.shape[0]))
		for i in range(filters.shape[0]):
			for j in range((inputobject.shape[0]-filters.shape[1])/stride[0]): #rows
				for k in range((inputobject.shape[1]-filters.shape[2])/stride[1]): #columns
					output[j,k,i] = (biases[i] + np.vdot(
						inputobject[np.ix_(np.arange(j*stride[0], j*stride[0] + filters.shape[1]), np.arange(k*stride[1], k*stride[1] + filters.shape[2]))],
						filters[i]))
		return output

	return convolute(zero_pad(inputobject, zero_pad_dimensions), filters, biases, stride)

#fully-connected layer
def fulcon_layer(inputobject, weights, biases):
	output = numpy.zeros((weights,))
	for i in output:
		output[i] = numpy.dot(inputobject.flatten(), weights[i]) + biases[i]
	return output

#returns the softmax of a 1d array
def softmax(array):
	return np.divide(np.exp(array), np.sum(np.exp(array)))

#relu activation function
def relu(array):
	array[array < 0] = 0
	return array


#BACKPROP

#calculate the error of the final layer
def final_layer_error():
	def d_loss_d_activation(predictions, labels):

#calculate the error of the layer lower to the last calculated layer
def lower_layer_error():

#cross entropy loss designed to work with one-hot labels and softmax predictions UNTESTED
def losscrossentropy(predictions, labels):
	output = 0
	for i in range(len(predictions)):
		output -= labels[i] * np.log(predictions[i])
	return output/i

#quadratic loss function for one-hot labels and softmax predictions
def lossquadratic(predictions, labels):
	return np.sum(np.subtract(labels, predictions))/(2^labels.size)

#print(relu(conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), init_weights(5, (2,2,3)), init_biases(5), zero_pad_dimensions=(2,2))))