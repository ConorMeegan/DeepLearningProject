import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import re

import model

def main():
    print("Deep Learning Project - Group 16"
          "\n--------------------------------"
          "\nConor Meegan - 16347531"
          "\nHugh Ormond - 16312941"
          "\nStephen Doughten - 19203845"
          "\n--------------------------------")

    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    tweets = pd.read_csv('training.csv', header=None, names=cols,
                         encoding='latin-1')  # unicode decode error if not latin-1

    x_data = tweets['text']
    y_label = tweets['target']

    # this is the preprocessed file being read in --> name it whatever you'd like e.g file.txt
    x_data_load_processed = open("full_preprocessed.txt", "r")
    x_data_processed = x_data_load_processed.readlines()

    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    text_tokenizer.fit_on_texts(x_data)
    text_tokenizer.fit_on_texts(x_data_processed)  # preprocessed
    tensor = text_tokenizer.texts_to_sequences(x_data)
    tensor_processed = text_tokenizer.texts_to_sequences(x_data_processed)  # preprocessed
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    tensor_processed = tf.keras.preprocessing.sequence.pad_sequences(tensor_processed, padding='post')  # preprocessed

    target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    text_labels = [str(x) for x in y_label]
    target_tokenizer.fit_on_texts(text_labels)
    targets = target_tokenizer.texts_to_sequences(text_labels)
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, padding='post')

    # doing a train test split of 70% training and 30% test
    x_train, x_test, y_train, y_test = train_test_split(tensor, targets, test_size=0.2, random_state=0)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(tensor_processed, targets, test_size=0.2, random_state=0)

    vocab_size = len(text_tokenizer.word_index) + 1

    # Create the model
    # rnn = model.Model(input_dim=vocab_size, dropout=0.5, epochs=1, batch_size=128, validation_split=0.2)
    # rnn.train_model(x_train, x_test, y_train, y_test)
    # rnn.train_model(x_train2, x_test2, y_train2, y_test2)


if __name__ == '__main__':
    main()
