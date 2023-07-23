"""
	Author : Aaryansh Sahay (@aaryansh) 
	Year : 2023
	Version : 3.0

"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow as tf


from keras.layers import (Dropout, Input, Dense, Conv2D,
						  MaxPooling2D, GlobalAveragePooling2D,
						  Upsampling2D, Conv2DTranspose, Reshape,
						  Flatten, Activation, BatchNormalization,
						  LeakyReLU, ReLU)

from tensorflow.keras.optimizers import Adam

from keras.models import Model


from keras.preprocessing import image
from keras.initializers import RandomNormal

from keras.utils.vis_utils import plot_model

import cv2
import random
import os
import time 
from utils import SaveImages, CheckpointCallback


class Generator:
	def __init__(self, seed_size, image_length, num_nodes):
		self.seed_size = seed_size
		self.image_length = image_length
		self.num_nodes = num_nodes

		init = RandomNormal(stddev = 0.2)

	def Build(self):
		seed_size, image_length, num_nodes = self.seed_size, self.image_length, self.num_nodes
		# number of Conv2D Transpose blocks based on the image dimension
		num_blocks = int(np.log2(image_length)) - 3 

		generator_input = Input(shape = (seed_size,))
		x = Dense(4*4*1024)(generator_input)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Reshape((4,4,1024))(x)

		x = Conv2DTranspose(num_nodes, kernel_size = 5, strides = 2, padding = "same",use_bias = False, kernel_initializer = init)(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		# Stacking Conv2D Blocks till target dimension is achieved 
		for i in range(num_blocks - 1):
			x = Conv2DTranspose(num_nodes, kernel_size = 5, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
			x = BatchNormalization()(x)
			x = ReLU()(x)

			# Second last block
			if i == (num_blocks - 1):
				x = Conv2DTranspose(128, kernel_size = 5, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
				x = BatchNormalization()(x)
				x = ReLU()(x)

		# Output block
		x = Conv2DTranspose(3, kernel_size = 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
		generator_output = layers.Activation(activations.tanh)(x)
		generator_model = Model(generator_input, generator_output)

		return generator_output

class Discriminator:
	def __init__(self, image_length, num_nodes, image_channels):
		self.image_length = image_length
		self.image_channels = image_channels
		self.num_nodes = num_nodes
		init = RandomNormal(stddev = 0.2)

	def Build(self):
		image_length,  image_channels, num_nodes = self.image_length, self.image_channels ,self.num_nodes
		
		discriminator_input = Input(shape = (image_length, image_length, image_channels))
		# Getting the number of conv2d blocks required to get to target image dimension
		num_blocks = int(np.log2(image_length)) - 3

		x = Con2D(64, kernel_size = 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
		x = LeakyReLU(alpha = 0.2)(x)

		for i in range(num_blocks -1):
			x = Conv2D(num_nodes, kernel_size = 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
			x = BatchNormalization()(x)
			x = LeakyReLU(alpha = 0.2)(x)

		# Last block
		x = Conv2D(1, kernel_size = 4, strides = 2, padding = "valid", use_bias = False, kernel_initializer = init)(x)
		x = Flatten()(x)

		discriminator_output = Dense(1, activation = "sigmoid")(x)
		discriminator_model  Model(discriminator_input, discriminator_output)

		return discriminator_model

class DCGAN(keras.Model):
	def __init__(self, seed_size, image_length, image_channels = 3, num_nodes = 512,model_optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5), **kwargs):
		super(DCGAN, self).__init__(**kwargs)
		self.seed_size = seed_size
		self.image_length = image_length
		self.num_nodes = num_nodes
		self.image_channels = image_channels
		self.model_optimizer = model_optimizer


		generator = Generator()
		generator_optimizer = model_optimizer
		generator_model = generator.Build(seed_size, image_length, num_nodes)
		self.generator = generator_model

		discriminator = Discriminator()
		discriminator_optimizer = model_optimizer
		self.discriminator = discriminator_model

		self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

	def GetPlot(self, **kwargs):
		return tf.keras.utils.plot_model(self.generator, show_shapes = True, **kwargs), tf.keras.utils.plot_model(self.discriminator, show_shapes = True, **kwargs)

	def GeneratorLoss(self, generator_pred):
		return self.cross_entropy(tf.ones_like(generator_pred), generator_pred)

	def DiscriminatorLoss(self, label, generator_pred, smooth = 0.1):
		label_loss = self.cross_entropy(tf.ones_like(label)*(1-smooth), label)
		generated_loss =self.cross_entropy(tf.zeros_like(generator_pred), generator_pred)
		total_loss = label_loss + generated_loss
		return total_loss

	def CompileModel(self, model_optimizer):
		model_optimizer = self.model_optimizer

		super(DCGAN, self).compile() 
		self.generator_optimizer = model_optimizer
		self.discriminator_optimizer = model_optimizer

	@tf.funciton
	def TrainStep(self, data):
		batch_size = tf.shape(data)[0]

		seed = tf.random.normal(shape = (batch_size, self.seed_size))
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_image = self.generator(seed, training = True)
			real_label = self.discriminator(data, training = True)
			fake_label = self.discriminator(generated_image, training = True)

			generator_loss = self.GeneratorLoss(fake_label)
			discriminator_loss = self.DiscriminatorLoss(real_label, fake_label)

			generator_gradient = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
			discriminator_gradient = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

			self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))
			self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))


		return {
			"generator loss" : generated_loss,
			"discriminator loss" : discriminator_loss
		}

