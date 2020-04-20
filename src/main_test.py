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
    tweets = pd.read_csv('../training.csv', header=None, names=cols,
                         encoding='latin-1')  # unicode decode error if not latin-1

    x_data = tweets['text']
    y_label = tweets['target']
	
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    text_tokenizer.fit_on_texts(x_data)
    tensor = text_tokenizer.texts_to_sequences(x_data)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
	
    target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    text_labels = [str(x) for x in y_label]
    target_tokenizer.fit_on_texts(text_labels)
    targets = target_tokenizer.texts_to_sequences(text_labels)
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, padding='post')
	
    # doing a train test split of 70% training and 30% test
    x_train, x_test, y_train, y_test = train_test_split(tensor, targets, test_size=0.3, random_state=0)
	
    '''num_rows=x_data.count()
    num_rows=int(num_rows[0])
    x_data2= x_data.iloc[0:0]
    x_test = x_data.head(10001).copy()
    for j in range(0, int(num_rows/1000)):
        first = j*1000
        last = (j+1)*1000-1
        x_test = x_data.iloc[first: last].copy()
        for i in range(first, last):
            #remove @s twitter only allows alphanumeric and underscores in their names
            x_test.loc[i, 'text'] = re.sub('@[A-Za-z0-9_]+|http[A-Za-z0-9_:/\.]+', '', str(x_test.loc[i, 'text']))
    x_data2.append(x_test)'''

    vocab_size = len(text_tokenizer.word_index) + 1

    # Create the model
    rnn = model.Model(input_dim=vocab_size, dropout=0.5, epochs=1, batch_size=128, validation_split=0.3)
    rnn.train_model(x_train, x_test, y_train, y_test)
	

if __name__ == '__main__':
    main()