#import tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
import re

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
    #print(tweets['text'].head(5))
    #test_text = tweets.loc[0, 'text']
    #print(test_text)
    #test_text = re.sub('@[A-Za-z0-9_]+', '', str(test_text))
    #print(test_text)
    num_rows=x_data.count()
    num_rows=int(num_rows[0])
    print(num_rows)
    x_data2= x_data.iloc[0:0]
    x_test = x_data.head(10001).copy()
    for j in range(0, int(num_rows/1000)):
        first = j*1000
        last = (j+1)*1000-1
        print(last)
        x_test = x_data.iloc[first: last].copy()
        for i in range(first, last):
            #remove @s twitter only allows alphanumeric and underscores in their names
            x_test.loc[i, 'text'] = re.sub('@[A-Za-z0-9_]+|http[A-Za-z0-9_:/\.]+', '', str(x_test.loc[i, 'text']))
            if i%1000 == 0:
                print(i)
    x_data2.append(x_test)


   # for i in x_test.index:
   #     #remove @s twitter only allows alphanumeric and underscores in their names
   #     x_test.loc[i, 'text'] = re.sub('@[A-Za-z0-9_]+', '', str(x_test.loc[i, 'text']))
   #     #remove all URLs, difficult to glean any meaning from them
   #     x_test.loc[i, 'text'] = re.sub('http[A-Za-z0-9_:/.]+', '', str(x_test.loc[i, 'text']))
   #     #print(x_data.loc[i, 'text'])
   #     if i%100 == 0:
   #         print(i)


    #x_data.to_csv('Training_xData')
    #y_label.to_csv('Training_yData')
 #   print(x_data.loc[500, 'text'])

if __name__ == '__main__':
    main()