import nltk
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd 
import re
import string
import preprocessing as processText

class ProcessText():	
	def clean_ats_and_links(self, text):
		#remove @s twitter only allows alphanumeric and underscores in their names
		return re.sub('@[A-Za-z0-9_]+|http[A-Za-z0-9_:/\.]+', '', str(text))
	
	def clean_digits(self, text):
		return re.sub('\d+', '', text)
		
	def remove_stopwords(self, text):
		stopword_list = stopwords.words('english')
		skip_words = ["no", "nor", "not"]
		
		words = text.split()
		cleaned_words = [word for word in words if (word not in stopword_list or word in skip_words) and len(word) > 1]
		
		return " ".join(cleaned_words)

	def remove_non_ascii(self, text):
		return re.sub('[^\x00-\x7F]+',  '', text)

	def to_lower(self, text):
		return text.lower()
	
if __name__ == '__main__':
	cols = ['target', 'id', 'date', 'flag', 'user', 'text']
	tweets = pd.read_csv('../training.csv', header=None, names=cols,
						encoding='latin-1')
							
	x_data = tweets['text']
	
	num_rows = x_data.count()
	num_rows = int(num_rows)
	x_data2 = x_data.iloc[0:0]
	x = x_data.head(10001).copy()
	processText = ProcessText()
	# print(x)
	output_file = open("output.txt", "w")

	for j in range(0, int(num_rows/1000)):
		first = j*1000
		last = (j+1)*1000-1
		x = x_data.iloc[first: last].copy()
		for i in range(first, last):
			# print(x.loc[i])
			# remove @s twitter only allows alphanumeric and underscores in their names
			x.loc[i] = processText.clean_ats_and_links(x.loc[i])
			x.loc[i] = processText.clean_digits(x.loc[i])
			x.loc[i] = processText.remove_stopwords(x.loc[i])
			x.loc[i] = processText.remove_non_ascii(x.loc[i])
			x.loc[i] = processText.to_lower(x.loc[i])
			output_file.write("%s\n" % x.loc[i])
			# print(x.loc[i])
			if i % 1000 == 0:
				print(i)
	x_data2.append(x)
