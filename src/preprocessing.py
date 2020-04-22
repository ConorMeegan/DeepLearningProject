import nltk
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd 
import re
import string


class ProcessText():	
	def clean_ats_and_links(text):
		#remove @s twitter only allows alphanumeric and underscores in their names
		return re.sub('@[A-Za-z0-9_]+|http[A-Za-z0-9_:/\.]+', '', str(text))
	
	def clean_digits(text):
		return re.sub('\d+', '', text)
		
	def remove_stopwords(text):
		stopword_list = stopwords.words('english')
		skip_words = ["no", "nor", "not"]
		
		words = text.split()
		cleaned_words = [word for word in words if (word not in stopword_list or word in skip_words) and len(word) > 1]
		
		return " ".join(cleaned_words)
		
	
if __name__ == '__main__':
	cols = ['target', 'id', 'date', 'flag', 'user', 'text']
	tweets = pd.read_csv('../training.csv', header=None, names=cols,
						encoding='latin-1')
							
	x_data = tweets['text']
	
	num_rows=x_data.count()
	num_rows=int(num_rows)
	x_data2= x_data.iloc[0:0]
	x = x_data.head(10001).copy()
	processText = ProcessText()
	for j in range(0, int(num_rows/1000)):
		first = j*1000
		last = (j+1)*1000-1
		x = x_data.iloc[first: last].copy()
		for i in range(first, last):
			#remove @s twitter only allows alphanumeric and underscores in their names
			x.loc[i, 'text'] = processText.clean_ats_and_links(x.loc[i, 'text'])
			x.loc[i, 'text'] = processText.clean_digits(x.loc[i, 'text'])
			x.loc[i, 'text'] = processText.remove_stopwords(x.loc[i, 'text'])
			if i%100 == 0:
				print(i)
			
	x_data2.append()