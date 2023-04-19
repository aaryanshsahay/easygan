import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, activations
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt

from keras.layers import (Dropout, Input, Dense, Conv2D,
						  MaxPooling2D, GlobalAveragePooling2D,
						  UpSampling2D, Conv2DTranspose,Reshape,
						  Flatten, Activation, BatchNormalization,
						  LeakyReLU, ReLU)



from keras.activations import tanh, sigmoid
from keras.losses import BinaryCrossentropy

from tensorflow.keras.optimizers import Adam

from keras.models import Model, Sequential
from keras.initializers import RandomNormal

from keras.utils.vis_utils import plot_model

import cv2

import random
import os
import time


class Generator:

	def __init__(self):
		#self.seed_size = seed_size
		#self.image_length = image_length
		pass

	def Build(self, seed_size, image_length, image_channels):
		init = tf.keras.initializers.RandomNormal(stddev = 0.02)
		numBlocks = int(np.log2(image_length)) - 3
		#print("Number of Blocks : ",numBlocks) 

		generatorInput = Input(shape = (seed_size,))
		x = Dense(4*4*1024)(generatorInput)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Reshape((4, 4, 1024))(x)

		x = Conv2DTranspose(512, kernel_size = 5, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		#print("Input Block and 1st Block initiaized")
		for i in range(numBlocks-1):
			x = Conv2DTranspose(512, kernel_size = 5, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
			x = BatchNormalization()(x)
			x = ReLU()(x)

 
			
			
			if i ==(numBlocks-1):
				x = Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = "same", use_bias = False, kernel_initalizer = init)(x)
				x = BatchNormaliztion()(x)
				x = ReLU()(x)


		x = Conv2DTranspose(image_channels, kernel_size = 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
		generatorOutput = layers.Activation(activations.tanh)(x)
		generatorModel = Model(generatorInput, generatorOutput)
		return generatorModel


class Discriminator:
	def __init__(self):
		pass

	def Build(self, image_length, image_channels):
		init = tf.keras.initializers.RandomNormal(stddev = 0.02)
		discriminatorInput = Input(shape = (image_length, image_length, image_channels))
		numBlocks = int(np.log2(image_length)) - 3
		
		x = Conv2D(64, kernel_size = 3, strides = 2, padding = "same", use_bias = False)(discriminatorInput)
		x = LeakyReLU(alpha = 0.2)(x)

		for i in range(numBlocks - 1):
			x = Conv2D(256, kernel_size = 3, strides = 2, padding = "same", use_bias = False, kernel_initializer = init)(x)
			x = BatchNormalization()(x)
			x = LeakyReLU(alpha = 0.2)(x)
		
		x = Conv2D(1, kernel_size = 4, strides = 2, padding = "valid", use_bias = False, kernel_initializer = init)(x)
		x = Flatten()(x)
		discriminatorOutput = Dense(1, activation = "sigmoid")(x)
		discriminatorModel = Model(discriminatorInput, discriminatorOutput)

		return discriminatorModel


class BinaryDCGAN(keras.Model):
	"""Subclass of the kears.Model class to define the custom training iteration and loss functions."""
	def __init__(self, seed_size, image_length, image_channels, **kwargs):
		
		super(BinaryDCGAN, self).__init__(**kwargs)
		self.seed_size = seed_size
		self.image_length = image_length
		self.image_channels = image_channels

		generator = Generator()
		gen_model = generator.Build(seed_size, image_length, image_channels)
		self.generator = gen_model

		discriminator = Discriminator()
		disc_model = discriminator.Build(image_length, image_channels)
		self.discriminator = disc_model

		self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

	def GetPlot(self, **kwargs):
		return tf.keras.utils.plot_model(self.generator, show_shapes=True, **kwargs) , tf.keras.utils.plot_model(self.discriminator,  show_shapes=True, **kwargs)



	def GeneratorLoss(self, fakeOutput):
		return self.cross_entropy(tf.ones_like(fakeOutput), fakeOutput)

	def DiscriminatorLoss(self, real_output, fake_output, smooth = 0.1):
		real_loss = self.cross_entropy(tf.ones_like(real_output)*(1-smooth), real_output)
		fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss

	def compile(self, generator_optimizer, discriminator_optimizer):
		super(BinaryDCGAN, self).compile()
		self.generator_optimizer = generator_optimizer
		self.discriminator_optimizer = discriminator_optimizer

	@tf.function
	def train_step(self, data):
		batch_size = tf.shape(data)[0]

		seed = tf.random.normal(shape = (batch_size, self.seed_size))
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generatedImage = self.generator(seed, training = True)
			real_output = self.discriminator(data, training = True)
			fake_output = self.discriminator(generatedImage, training = True)

			genLoss = self.GeneratorLoss(fake_output)
			discLoss = self.DiscriminatorLoss(real_output, fake_output)

			generatorGrad = gen_tape.gradient(genLoss, self.generator.trainable_variables)
			discriminatorGrad = disc_tape.gradient(discLoss, self.discriminator.trainable_variables)

			self.generator_optimizer.apply_gradients(zip(generatorGrad, self.generator.trainable_variables))
			self.discriminator_optimizer.apply_gradients(zip(discriminatorGrad, self.discriminator.trainable_variables))


		return {
				"generator loss" : genLoss,
				"discriminator loss" : discLoss
				}


class SaveEpochs(keras.callbacks.Callback):
	def __init__(self, seed_size, **kwargs):
		super(keras.callbacks.Callback, self).__init__(**kwargs)
		
		self.noise = tf.random.normal(shape=(4 * 7, seed_size))
		self.seed_size = seed_size
		self.margin = 16
		self.num_rows = 4
		self.num_cols = 7

	def on_epoch_end(self, epoch, logs = None):
		imageArray = np.full((
							self.margin + (self.num_rows * (64 + self.margin)),
							self.margin + (self.num_cols * (64 + self.margin)),
							3), 255, dtype = np.uint8)

		generatedImages = self.model.generator.predict(self.noise)
		generatedImages = generatedImages * 0.5 + 0.5


		imageCount = 0
		for row in range(self.num_rows):
			for col in range(self.num_cols):
				r = row * (64 + 16) + self.margin
				c = col * (64 + 16) + self.margin
				imageArray[r:r + 64, c:c + 64] = generatedImages[imageCount] * 255
				imageCount +=1



		outputPath = "EpochImages"

		if not os.path.exists(outputPath):
			os.makedirs(outputPath)


		filename = os.path.join(outputPath, f"train-epoch-{epoch+1}.png")
		im = Image.fromarray(imageArray)
		im.save(filename)


class CheckpointCallback(keras.callbacks.Callback):
	def __init__(self, **kwargs):
		super(keras.callbacks.Callback, self).__init__(**kwargs)

	def on_epoch_end(self, epoch, logs = None):
		outputPath = "Weights"

		if not os.path.exists(outputPath):
			os.makedirs(outputPath)

		self.model.generator.save_weights("Weights/generator_weights.h5")
		self.model.discriminator.save_weights("Weights/discriminator_weights.h5")
