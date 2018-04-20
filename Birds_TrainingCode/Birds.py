# Copyright (c) 2018/4/17 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.


# Retrain Inception V3 for Birds CUB-200-2011 Dataset.
# Size of the feature vector that are output by Inception V3 is 2048.


#######################################
# Network Architecture
# input layer(feature vector)
# output layer(logits layer)
#######################################


#######################################
# Usage: 
# Accuracy of the model on test set is 48.02%.
# Run this code to test the accuracy of our model.
# If you want to retrain the model, uncomment the training code in the main function and set the 'Steps' & 'Loops' global variables to control the training process.
#######################################


#################### Libs ####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import datetime


#################### Global Variables ####################
# the params of the input dataset.
# the width, height & channels of the images in the input TFRecords file.
ImageWidth = 299
ImageHeight = 299
ImageChannels = 3
# the number of categories to classify.
Class_Count = 200
# the size of the input feature vector.
Feature_Vector_Size = 2048
# the size of the buffer for shuffling.
# buffer_size should be greater than the number of examples in the Dataset, ensuring that the data is completely shuffled.
Buffer_Size = 100000


#################### Hyper Parameters of the Model #######################
# the batch size for training.
Batch_Size = 100
# training steps in a loop.
Training_Steps = 200
# Loops is the total number of training times.
# Total_Training_Steps = Steps* Loops.
Loops = 3
# the dropout rate of the feature vector layer.
Dropout_Rate = 0.3
# the learning rate of the optimizer.
Learning_Rate = 0.1


#################### Tensorflow Settings. ####################
# Output the logging info.
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode):
	"""the model function for CNN OxFlower."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# oxflower images are 224*224 pixels, and have 3 color channels
	# print(features)
	input_layer = features['feature_vector']
	
	# Add dropout operation; 0.6 probability that element will be kept
	dropout_1 = tf.layers.dropout(inputs = input_layer, rate = Dropout_Rate, training=mode == tf.estimator.ModeKeys.TRAIN)
	# print(dropout_1)
	
	# Logits layer
	# Output Tensor Shape: [batch_size, CategoryNum]
	# Default: activation=None, maintaining a linear activation.
	logits = tf.layers.dense(inputs = dropout_1, units = Class_Count)
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
		optimizer = tf.train.AdamOptimizer(Learning_Rate)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)  
	

def parse_function(example_proto):
	"""parse function is used to parse a single TFRecord example in the dataset."""
	# Parses a single Example proto.
	# Returns a dict mapping feature keys to Tensor and SparseTensor values.
	features = tf.parse_single_example(example_proto,features={
	'label': tf.FixedLenFeature([], tf.int64), 'feature_vector' : tf.FixedLenFeature(shape = (Feature_Vector_Size,), dtype = tf.float32),})
	feature_vector = tf.cast(features['feature_vector'], tf.float32)
	labels = tf.cast(features['label'], tf.int64) 
	return {'feature_vector': feature_vector}, labels

	
def train_input_fn(tfrecords, batch_size):
	"""
	An input function for training mode.
	tfrecords: the filename of the training TFRecord file, batch_size: the batch size.
	"""
	# read the TFRecord file into a dataset.
	dataset = tf.data.TFRecordDataset(tfrecords)
	# parse the dataset.
	dataset = dataset.map(parse_function)
	# the size of the buffer for shuffling.
	# buffer_size should be greater than the number of examples in the Dataset, ensuring that the data is completely shuffled. 
	buffer_size = Buffer_Size
	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)
	# print(dataset)
	
	# make an one shot iterator to get the data of a batch.
	train_iterator = dataset.make_one_shot_iterator()
	# get the features and labels.
	features, labels = train_iterator.get_next()
	# print(features)
	
	# print(labels)
	return features, labels

def eval_input_fn(tfrecords, batch_size):
	"""
	An input function for evaluation mode.
	tfrecords: the filename of the evaluation/test TFRecord file, batch_size: the batch size.
	"""
	# read the TFRecord file into a dataset.
	dataset = tf.data.TFRecordDataset(tfrecords)
	# parse the dataset.
	dataset = dataset.map(parse_function)
	
	# Shuffle, repeat, and batch the examples.
	dataset = dataset.batch(batch_size)
	# print(dataset)
		# make an one shot iterator to get the data of a batch.
	eval_iterator = dataset.make_one_shot_iterator()
	# get the features and labels.
	features, labels = eval_iterator.get_next()
	# print(features)
	# print(labels)
	return features, labels


def main(unused_argv):
	
	# Create the Estimator
	birds_classifier = tf.estimator.Estimator(
		model_fn = model_fn,
		model_dir = "Models/Feature_Vector/")
	
	
	"""
	# Uncomment this to retain the model.
	# train and validate the model in a loop.
	# the start time of training.
	start_time = datetime.datetime.now()
	for i in range(Loops):
		# Train the model
		# input('1:')
		birds_classifier.train(
			input_fn=lambda:train_input_fn('Dataset/trainBottlenecks.tfrecords', Batch_Size),
			steps = Training_Steps)
		
		# input('2:')
		# Evaluate the model on validation set.
		eval_results = birds_classifier.evaluate(input_fn=lambda:eval_input_fn('Dataset/validationBottlenecks.tfrecords', Batch_Size))
		# Calculate the accuracy of our CNN model.
		accuracy = eval_results['accuracy']*100
		# print('\n\ntraining steps: {}'.format((i+1)*Steps))
		print('\n\nValidation Set accuracy: {:0.2f}%\n\n'.format(accuracy))
		
	
	# the end time of training.
	end_time = datetime.datetime.now()
	print('\n\n\nTraining starts at: {}'.format(start_time))
	print('\nTraining ends at: {}\n\n\n'.format(end_time))
	"""
	
	# evaluate the model on test set.
	# Evaluate the model on test set.
	eval_results = birds_classifier.evaluate(input_fn=lambda:eval_input_fn('Dataset/testBottlenecks.tfrecords', Batch_Size))
	# Calculate the accuracy of our CNN model.
	accuracy = eval_results['accuracy']*100
	print('\n\nTest Set accuracy: {:0.2f}%\n\n'.format(accuracy))	
	
if __name__ == "__main__":
	"""tf.app.run() runs the main function in this module by default."""
	tf.app.run()
