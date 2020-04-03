import tensorflow as tf
import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

MODEL_FILE_PATH = './/models//RNNModel.h5'

class Model():
	# input_dim: Size of input into the embedding layer. Should be the size of the vocabulary
	# dropout: Fraction of units to drop
	# epochs: Number of times the dataset is passed through
	# batch_size: Size of each batch into which the data is divided
	# train_split: The fraction of the data which will be used for the training set, with the 
	#			  remainder used for the test set. 
	# validation_split: Fraction of the traning data used for the Validation Set
	def __init__(self, input_dim, dropout, epochs, batch_size, train_split, validation_split):
		self.input_dim = input_dim
		self.dropout = dropout
		self.epochs = epochs
		self.batch_size = batch_size
		self.train_split = train_split	
		self.validation_split = validation_split
	
	def build_model(self):
		model = models.Sequential()
		# Number of layers and layer sizes will likely need a lot of tweaking
		model.add(layers.Embedding(self.input_dim, output_dim=64))
		model.add(layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
		model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
		model.add(layers.Bidirectional(layers.LSTM(32)))
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dropout(self.dropout))
		model.add(layers.Dense(1, activation='sigmoid'))		
		model.summary()
		
		# Loss function and optimizer may need to be changed, I havn't looked into them in detail
		model.compile(loss='binary_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])
						
		return model
												
	def plot_graphs(history, string):
		plt.plot(history.history[string])
		plt.plot(history.history['val_'+string], '')
		plt.xlabel("Epochs")
		plt.ylabel(string)
		plt.legend([string, 'val_'+string])
		plt.show()				
		
	def split_data(self, data, target):
		train_end = int(self.train_split * len(data))
		train_data = data[:train_end]
		test_data = data[train_end:]
		
		train_target = target[:train_end]
		test_target = target[train_end:]
		
		return train_data, test_data, train_target, test_target
					
	# This is the function which should be called once the class is created
	# Will split up the data, create the model, then train and evaluate it
	#
	# data: The list of input 
	def train_model(self, data, target):
		train_data, test_data, train_target, test_target = self.split_data(data, target)
	
		model = self.build_model()
	
		history = model.fit(train_data, train_target,
				batch_size=self.batch_size,
				validation_split =self.validation_split,
				epochs=self.epochs)
				
		model.save_weights(MODEL_FILE_PATH)
	
		print('\nhistory dict:', history.history)
	
		plot_graphs(history, "loss")
	
		print('\n# Evaluate on test data')
		results = model.evaluate(test_data, test_target)
		print('test loss, test acc:', results)

