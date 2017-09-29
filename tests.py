#Tests Module

import numpy as np

import backprop
import import_files
import layer
import tests

def old_tests():
	print(backprop.final_layer_error(layer.softmax(np.array([5,2,5,1])), np.array([0,1,0,0]), np.array([5,2,5,1])))
	print(layer.conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), layer.init_weights(5, (2,2,3)), layer.init_biases(5), zero_pad_dimensions=(2,2)))
	print(layer.conv_layer(
		layer.relu(
			layer.conv_layer(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]), layer.init_weights(5, (2,2,3)), layer.init_biases(5), zero_pad_dimensions=(2,2))),
		layer.init_weights(32, (2,2,5)),
		layer.init_biases(32),
		zero_pad_dimensions=(1,1)
		).shape)
	import_files.import_batch("/Users/admin/Documents/code/python/tensorflow/projects/CIFAR-10-convnet/Data/test/", 1, 256)


def test_full_net():
	"""
	test whether an evaluable network can be created
	"""

	#import

	test_data = import_files.import_batch("/Users/admin/Documents/code/python/tensorflow/projects/CIFAR-10-convnet/Data/test/", 1, 10)

	#weights and biases

	layer1_weights = layer.init_weights(32, (4,4,3))
	layer1_biases = layer.init_biases(32)

	layer2_weights = layer.init_weights(128, (8192))
	layer2_biases = layer.init_biases(128)

	layer3_weights = layer.init_weights(10, (128))
	layer3_biases = layer.init_biases(10)

	output = np.empty((10, 10))
	for i in range(0,9):
		layer1 = layer.relu(layer.conv_layer(test_data[i], layer1_weights, layer1_biases, zero_pad_dimensions=(2,2), stride=(2,2)))
		layer2 = layer.relu(layer.fulcon_layer(layer1, layer2_weights, layer2_biases))
		layer3 = layer.relu(layer.fulcon_layer(layer2, layer3_weights, layer3_biases))
		print(layer3)
		output[i] = layer.softmax(layer3)