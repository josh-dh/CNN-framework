#Backprop Module

import numpy as np

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


def stochastic_gradient_descent(batch, parameters, parameter_derivatives, step_size):
	"""
	single stochastic gradient descent iteration for parameters performed over a batch
	"""
	pass