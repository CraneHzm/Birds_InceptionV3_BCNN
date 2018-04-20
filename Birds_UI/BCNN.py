# Copyright (c) 2018/4/19 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.


# Retrain Inception V3 using Bilinear CNN for Birds CUB-200-2011 Dataset.
# Size of the feature vector that are output by Inception V3 is 2048.


#######################################
# Network Architecture
# input layer(feature vector)
# Bilinear Layer(The 2 feature vector layers which are combined by the Bilinear Layer are identical, you can also combine 2 different convolution layers)
# output layer(logits layer)
#######################################


#################### Libs ####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class BCNN(object):
	def __init__(self):
		# member variables
		# the number of categories to classify.
		self.classCount = 200
		# the required size of the input feature vector.
		self.feature_vector_size = 2048
		# create the classifier.
		self.classifier = tf.estimator.Estimator(model_fn = self.model_fn, model_dir = "Models/BCNN/")

		
	def predict(self, bottleneck_value):
		'''predict the class of the given bottleneck value'''
		# bottleneck_value is expected to be of size [1, 2048]
		# assert the input vector's size.
		if self.feature_vector_size != bottleneck_value.shape[1]:
			print('Size of the Input feature vector must be: [1, 2048]'.format(self.feature_vector_size))
			return -1
		else:
			print('Input Bottleneck Values Received!\n\n')

		# when you use numpy ndarray as your input, you can use numpy_input_fn to create the input function.
		pred_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": bottleneck_value}, shuffle=False)
		
		predictions = self.classifier.predict(input_fn = pred_input_fn)
		predictions = next(predictions)
	
		# ID of the class.
		classID = predictions['classes'] + 1
		return classID
		
		
	def prd_input_fn(self, feature):
		"""An input function for prediction."""
		# No labels, use only features.
		# Convert the inputs to a tensor Data.
		data = tf.convert_to_tensor(feature)
		# print(data)
		data = tf.reshape(data, [1, data.shape[0]])
		# print(data)
		# input('input')
		# dataset = dataset.batch(batch_size)
		dataset = tf.data.Dataset.from_tensor_slices(data)
		batch_size = 1
		dataset = dataset.batch(batch_size)
		return dataset
		
		
	def model_fn(self, features, labels, mode):
		"""the model function for CNN OxFlower."""
		# Input Layer
		# Reshape X to 4-D tensor: [batch_size, width, height, channels]
		# print(features)
		input_layer = features['x']
		# print(input_layer)
		# input('input')
		
		
		# input_layer = tf.layers.batch_normalization(inputs = input_layer)
		# The bilinear layer.
		# We combine the 2 identical feature vector layers in our code. You can also combine 2 different convolution layers.
		# The bilinear layer is connected to the final output layer(the logits layer).
		phi_I = tf.einsum('im,in->imn', input_layer, input_layer)
		# print(phi_I)
		# input('input')
		
		
		# phi_I = tf.reshape(phi_I, [-1,Feature_Vector_Size*Feature_Vector_Size])
		phi_I = tf.reduce_max(phi_I, axis=2, keepdims=False)
		#print(phi_I)
		# print(phi_I)
		phi_I = tf.layers.batch_normalization(inputs = phi_I)
		# print(phi_I)
		# input("input: ")
		
		# Fully Connected Dense Layer #1
		# fc_1 = tf.layers.dense(inputs = phi_I, units = 512, activation=tf.nn.relu)

		dropout_1 = tf.layers.dropout(inputs = phi_I, rate = 0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
		# print(dropout_1)
		
		# Logits layer
		# Output Tensor Shape: [batch_size, Class_Count]
		# Default: activation=None, maintaining a linear activation.
		logits = tf.layers.dense(inputs = dropout_1, units = self.classCount)
		# print(logits)
		
		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
		
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

		# Calculate Loss (for both TRAIN and EVAL modes)
		# No need to use one-hot labels.
		loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
		

		# Calculate evaluation metrics.
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='acc_op')
		eval_metric_ops = {'accuracy': accuracy}
		# Use tensorboard --logdir=PATH to view the graphs.
		# The tf.summary.scalar will make accuracy available to TensorBoard in both TRAIN and EVAL modes. 
		tf.summary.scalar('accuracy', accuracy[1])
		
		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.AdamOptimizer(0.001)
			train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)  

			
if __name__ == "__main__":
	bcnn = BCNN()
	print('Required Input Feature Vector Size: {}'.format(bcnn.feature_vector_size))
	print('The Number of Categories to be Classified: {}'.format(bcnn.classCount))
	bottleneck_value = np.random.rand(1, 2048)
	bottleneck_value = bottleneck_value.astype('float32')
	print('Input Bottleneck Values: ')
	# print(bottleneck_value.shape[0])
	print(bottleneck_value.dtype)
	print(bottleneck_value.shape)
	classID = bcnn.predict(bottleneck_value)
	print('\n\n\nClass ID： {}'.format(classID))
