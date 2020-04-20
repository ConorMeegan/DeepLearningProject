import nltk
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd 
import re
import string


def main():
	cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    tweets = pd.read_csv('training.csv', header=None, names=cols,
                         encoding='latin-1')
						 
	text = tweets.drop('target', axis=1)
	
	
def ProcessText(text):
	def Clean_ats_and_link(text):
		#remove @s twitter only allows alphanumeric and underscores in their names
		return re.sub('@[A-Za-z0-9_]+|http[A-Za-z0-9_:/\.]+', '', str(text))
	
	def clean_digits(text):
		return re.sub('\d+', '', input_text)
		
	def remove_stopwords(text):
		stopword_list = stopwords.words('english')
		skip_words = ["no", "nor", "not"]
		
		words = text.split()
		cleaned_words = [word for word in words if (word not in stopword_list or word in skip_words) and len(word) > 1]
		
		return " ".join(cleaned_words)