# Copyright (c) 2018/4/16 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.

# Calculate the Bottleneck tensors of the original input data and save the Bottlenecks to TFRecord files.


#################### Libs. ####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


#################### Global Variables. ####################
# the URL of the pre-trained model.
HUB_MODULE = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
# the model spec.
Module_Spec = hub.load_module_spec(HUB_MODULE)
# the image size that is required by this model.
Module_Height, Module_Width = hub.get_expected_image_size(Module_Spec)
Modelu_Depth = hub.get_num_image_channels(Module_Spec)
# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')
				  
# the size of our input images.
ImageHeight = Module_Height
ImageWidth = Module_Width
ImageChannels = Modelu_Depth


#################### Tensorflow Settings. ####################
# Output the logging info.
tf.logging.set_verbosity(tf.logging.INFO)



def parse_function(example_proto):
	"""parse function is used to parse a single TFRecord example in the dataset."""
	# Parses a single Example proto.
	# Returns a dict mapping feature keys to Tensor and SparseTensor values.
	features = tf.parse_single_example(example_proto,features={
	'label': tf.FixedLenFeature([], tf.int64), 'img_raw' : tf.FixedLenFeature([], tf.string),})
	# Reinterpret the bytes of a string as a vector of numbers.
	imgs = tf.decode_raw(features['img_raw'], tf.uint8)
	# Reshapes a tensor.
	# [1, ImageWidth, ImageHeight, ImageChannels] means each image data is used as the input tensors of Hub Module([none, height, width, 3]).
	imgs = tf.reshape(imgs, [1, ImageWidth, ImageHeight, ImageChannels])  
	# cast the data from (0, 255) to (0, 1).
	# (0, 1) range is required by Hub Module.
	imgs = tf.cast(imgs, tf.float32) * (1. / 255)
	labels = tf.cast(features['label'], tf.int64) 
	return {'x': imgs}, labels


def read_tfrecords(tfrecords):
	"""
	read the tfrecord files.
	tfrecords: the filename of the TFRecord file.
	"""
	# read the TFRecord file into a dataset.
	dataset = tf.data.TFRecordDataset(tfrecords)
	# parse the dataset.
	dataset = dataset.map(parse_function)
	
	# make an one shot iterator to get the data of a batch.
	train_iterator = dataset.make_one_shot_iterator()
	# get the features and labels.
	features, labels = train_iterator.get_next()
	# print(features)
	
	# print(labels)
	return features, labels


def parse_function2(example_proto):
	"""parse function is used to parse a single TFRecord example in the dataset."""
	# Parses a single Example proto.
	# Returns a dict mapping feature keys to Tensor and SparseTensor values.
	features = tf.parse_single_example(example_proto,features={
	'label': tf.FixedLenFeature([], tf.int64), 'feature_vector' : tf.FixedLenFeature(shape = (2048), dtype = tf.float32),})
	feature_vector = tf.cast(features['feature_vector'], tf.float32)
	labels = tf.cast(features['label'], tf.int64) 
	return {'feature_vector': feature_vector}, labels


def read_tfrecords2(tfrecords):
	"""
	read the tfrecord files.
	tfrecords: the filename of the TFRecord file.
	"""
	# read the TFRecord file into a dataset.
	dataset = tf.data.TFRecordDataset(tfrecords)
	# parse the dataset.
	dataset = dataset.map(parse_function2)
	
	# make an one shot iterator to get the data of a batch.
	train_iterator = dataset.make_one_shot_iterator()
	# get the features and labels.
	features, labels = train_iterator.get_next()
	# print(features)
	
	# print(labels)
	return features, labels
	
	
def create_module_graph(module_spec):
	"""
	Creates a graph and loads Hub Module into it.

	Args:
		module_spec: the hub.ModuleSpec for the image module being used.

	Returns:
		graph: the tf.Graph that was created.
		bottleneck_tensor: the bottleneck values output by the module.
		resized_input_tensor: the input images, resized as expected by the module.
		wants_quantization: a boolean, whether the module has been instrumented with fake quantization ops.
	"""
	
	with tf.Graph().as_default() as graph:
		resized_input_tensor = tf.placeholder(tf.float32, [None, ImageHeight, ImageWidth, 3])
		m = hub.Module(module_spec)
		bottleneck_tensor = m(resized_input_tensor)
		wants_quantization = any(node.op in FAKE_QUANT_OPS for node in graph.as_graph_def().node)
	return graph, bottleneck_tensor, resized_input_tensor, wants_quantization

	
def main(unused_argv):
	
	# Set up the pre-trained graph.
	graph, bottleneck_tensor, resized_input_tensor, wants_quantization = create_module_graph(Module_Spec)
	
	# Add variables to the graph.
	with graph.as_default():
		# read the original input data set.
		features_train, labels_train = read_tfrecords('train_aug.tfrecords')
		features_validation, labels_validation = read_tfrecords('validation.tfrecords')
		features_test, labels_test = read_tfrecords('test.tfrecords')
		# save the bottleneck values of the corresponding input data into TFRecord files.
		train_writer_2048 = tf.python_io.TFRecordWriter("../Dataset/trainBottlenecks.tfrecords")
		validation_writer_2048 = tf.python_io.TFRecordWriter("../Dataset/validationBottlenecks.tfrecords")
		test_writer_2048 = tf.python_io.TFRecordWriter("../Dataset/testBottlenecks.tfrecords")
		# train_writer_512 = tf.python_io.TFRecordWriter("../Dataset/trainBottlenecks512.tfrecords")
		# validation_writer_512 = tf.python_io.TFRecordWriter("../Dataset/validationBottlenecks512.tfrecords")
		# test_writer_512 = tf.python_io.TFRecordWriter("../Dataset/testBottlenecks512.tfrecords")		
	
	# Run the Session on our graph.
	with tf.Session(graph=graph) as sess:
		# Initialize all weights: for the module to their pretrained values,
		# and for the newly added retraining layer to random initial values.	
		init = tf.global_variables_initializer()
		sess.run(init)
		
		
		# read the original train data, calculate the corresponding bottlenecks and save them into TFRecord files.
		# the number of examples.
		num_example = 0
		while True:
			try:
				# read the resized input values.
				resized_input_values = sess.run(features_train['x'])
				# read the labels.
				label_value = int(sess.run(labels_train))
			except tf.errors.OutOfRangeError:
				print("End of train dataset.")
				break
			else:
				# Run Hub Module on the input data and calculate the bottleneck values.
				bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
				# (1, 2048) --> (2048, )
				bottleneck_values = np.squeeze(bottleneck_values)	
				# print(bottleneck_values.shape)
				# print(bottleneck_values[3])
				# The feature vector of size 512.
				bottleneck_values_512 = np.zeros(512)
				for i in range(512):
					bottleneck_values_512[i] = (bottleneck_values[i*4] + bottleneck_values[i*4 +1] + bottleneck_values[i*4 +2] + bottleneck_values[i*4 +3])/4.0
				# print(bottleneck_values_512.shape)	
				# input('input: ')
				# the example with feature vector of size 2048.
				example_2048 = tf.train.Example(features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label_value])),  
				'feature_vector': tf.train.Feature(float_list = tf.train.FloatList(value= bottleneck_values))})) 
				# the example with feature vector of size 512.
				example_512 = tf.train.Example(features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label_value])),  
				'feature_vector': tf.train.Feature(float_list = tf.train.FloatList(value= bottleneck_values_512))})) 				
				
				num_example = num_example + 1 
				print('Writing Train Example: {}'.format(num_example))
				train_writer_2048.write(example_2048.SerializeToString())
				# train_writer_512.write(example_512.SerializeToString())
		# close the writer.
		train_writer_2048.close()
		# train_writer_512.close()

		# read the original validation data, calculate the corresponding bottlenecks and save them into TFRecord files.
		# the number of examples.
		num_example = 0
		while True:
			try:
				# read the resized input values.
				resized_input_values = sess.run(features_validation['x'])
				# read the labels.
				label_value = int(sess.run(labels_validation))
			except tf.errors.OutOfRangeError:
				print("End of validation dataset.")
				break
			else:
				# Run Hub Module on the input data and calculate the bottleneck values.
				bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
				# (1, 2048) --> (2048, )
				bottleneck_values = np.squeeze(bottleneck_values)	
				# The feature vector of size 512.
				bottleneck_values_512 = np.zeros(512)
				for i in range(512):
					bottleneck_values_512[i] = (bottleneck_values[i*4] + bottleneck_values[i*4 +1] + bottleneck_values[i*4 +2] + bottleneck_values[i*4 +3])/4.0
				# the example with feature vector of size 2048.
				example_2048 = tf.train.Example(features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label_value])),  
				'feature_vector': tf.train.Feature(float_list = tf.train.FloatList(value= bottleneck_values))})) 
				# the example with feature vector of size 512.
				example_512 = tf.train.Example(features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label_value])),  
				'feature_vector': tf.train.Feature(float_list = tf.train.FloatList(value= bottleneck_values_512))}))
				
				num_example = num_example + 1 
				print('Writing Validation Example: {}'.format(num_example))
				validation_writer_2048.write(example_2048.SerializeToString())
				# validation_writer_512.write(example_512.SerializeToString())
				
		# close the writer.
		validation_writer_2048.close()
		# validation_writer_512.close()
		
		# read the original test data, calculate the corresponding bottlenecks and save them into TFRecord files.
		# the number of examples.
		num_example = 0
		while True:
			try:
				# read the resized input values.
				resized_input_values = sess.run(features_test['x'])
				# read the labels.
				label_value = int(sess.run(labels_test))
			except tf.errors.OutOfRangeError:
				print("End of Test dataset.")
				break
			else:
				# Run Hub Module on the input data and calculate the bottleneck values.
				bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
				# (1, 2048) --> (2048, )
				bottleneck_values = np.squeeze(bottleneck_values)	
				# The feature vector of size 512.
				bottleneck_values_512 = np.zeros(512)
				for i in range(512):
					bottleneck_values_512[i] = (bottleneck_values[i*4] + bottleneck_values[i*4 +1] + bottleneck_values[i*4 +2] + bottleneck_values[i*4 +3])/4.0
				# example defines the data format.
				example_2048 = tf.train.Example(features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label_value])),  
				'feature_vector': tf.train.Feature(float_list = tf.train.FloatList(value= bottleneck_values))})) 
				# the example with feature vector of size 512.
				example_512 = tf.train.Example(features=tf.train.Features(feature={  
				"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label_value])),  
				'feature_vector': tf.train.Feature(float_list = tf.train.FloatList(value= bottleneck_values_512))}))
				
				num_example = num_example + 1 
				print('Writing Test Example: {}'.format(num_example))
				test_writer_2048.write(example_2048.SerializeToString())
				# test_writer_512.write(example_512.SerializeToString())
		# close the writer.
		test_writer_2048.close()
		# test_writer_512.close()
		
if __name__ == "__main__":
	"""tf.app.run() runs the main function in this module by default."""
	tf.app.run()
