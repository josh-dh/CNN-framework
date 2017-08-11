import numpy as np

#convlayer

#input shape: (rows, columns, color: 3 if rgb, 1 if greyscale)

def conv_layer(input, filter_count, filter_dimensions, zero_pad_dimensions, train=false, init=false):
	#initialize filters with gaussian noise
	def init_filters(filter_count, filter_dimensions):
		filters = np.array([])
		for i in range(filter_count):
			np.append(filter, np.random.normal(scale=50,size=filter_dimensions), axis=0)
		return filters
	#zero pad equally on x and y axis equally per axis
	def zero_pad(input, zero_pad_dimensions):
		return np.append(
			np.zeros((input.shape()[0],zero_pad_dimensions[1])),
			np.append(np.zeros((zero_pad_dimensions[0],(input.shape()[1]))),input,np.zeros((zero_pad_dimensions[0],(input.shape()[1]))), axis=1),
			np.zeros((input.shape()[0],zero_pad_dimensions[1]))