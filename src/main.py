import tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    print("Deep Learning Project - Group 16"
          "\n--------------------------------"
          "\nConor Meegan - 16347531 "
          "\nHugh Ormond - 16312941"
          "\nStephen Doughten - 19203845"
          "\n--------------------------------")

    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    tweets = pd.read_csv('training.csv', header=None, names=cols,
                         encoding='latin-1')  # unicode decode error if not latin-1

    # Dropping the target label as this is what we will be learning
    x_data = tweets.drop('target', axis=1)
    y_label = tweets['target']
    # doing a train test split of 70% training and 30% test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=0)


if __name__ == '__main__':
    main()
