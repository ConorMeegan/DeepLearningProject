import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

MODEL_FILE_PATH = './/models//RNNModel.h5'


class Model ():
	# input_dim: Size of input into the embedding layer. Should be the size of the vocabulary
	# dropout: Fraction of units to drop
	# epochs: Number of times the dataset is passed through
	# batch_size: Size of each batch into which the data is divided
	# validation_split: Fraction of the traning data used for the Validation Set
	def __init__(self, input_dim, dropout, epochs, batch_size, validation_split):
		print("#### Creating Model...")
		self.input_dim = input_dim
		self.dropout = dropout
		self.epochs = epochs
		self.batch_size = batch_size
		self.validation_split = validation_split
	
	def build_model(self):
		model = Sequential()
		# Number of layers and layer sizes will likely need a lot of tweaking
		model.add(Embedding(self.input_dim, output_dim=64))
		model.add(Bidirectional(LSTM(64, return_sequences=True)))
		# model.add(Bidirectional(LSTM(64, return_sequences=True)))
		model.add(Bidirectional(LSTM(32))),
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(self.dropout))
		model.add(Dense(1))
		model.summary()

		optimizer = Adam(0.001, 0.9)
		# Loss function and optimizer may need to be changed, I havn't looked into them in detail
		model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
						
		return model
												
	def plot_graphs(history, string):
		plt.plot(history.history[string])
		plt.plot(history.history['val_'+string], '')
		plt.xlabel("Epochs")
		plt.ylabel(string)
		plt.legend([string, 'val_'+string])
		plt.show()				
					
	# This is the function which should be called once the class is created
	# Will create the model, then train and evaluate it
	def train_model(self, train_data, test_data, train_target, test_target):
		print("#### Begining model training...")
		model = self.build_model()
	
		history = model.fit(train_data, train_target,
				batch_size=self.batch_size,
				validation_split=self.validation_split,
				epochs=self.epochs)
				
		print("#### Training Complete. Saving weights")		
		model.save_weights(MODEL_FILE_PATH)
	
		print('\n####History dict:', history.history)
	
		plot_graphs(history, "loss")
	
		print('\n#### Evaluate on test data')
		results = model.evaluate(test_data, test_target)
		print('test loss, test acc:', results)

