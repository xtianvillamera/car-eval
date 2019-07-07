#!usr/bin/env python3
import os
import numpy as np
import string
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords


def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


sentiments = np.array([0,1])
features = []
target = []

folder = 'sentiment labelled sentences'
for filename in os.listdir(folder):
	if filename.endswith('.txt') and filename != 'readme.txt':
		#print(filename)
		full_filename = os.path.join(folder,filename)
		f = open(full_filename,'r')
		#print(f.read())
		with f as reading:
			line = reading.readlines()
		line = [x.strip() for x in line]
		line = [x.split('\t') for x in line]
		#print(line)
		for review in line:
			#print(review[1])
			features.append(review[0])
			target.append(review[1])
		#print(features)
		#print(target)
		f.close()	
	else:
		continue

features_no_sw = []
for rev in features:
	stringh = " ".join(text_process(rev))
	features_no_sw.append(stringh)
	
#print(features_no_sw)
#print(stringh)


target_arr = np.array(target)		
x_train, x_test, y_train, y_test = train_test_split(features, target_arr, test_size = 0.2, random_state=42)

text_clf = Pipeline([('vect',CountVectorizer()),('tf', TfidfTransformer()),('clf',MultinomialNB())])
#text_clf = Pipeline([('vect',CountVectorizer()),('tf', TfidfTransformer()),('clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None))])
text_clf.fit(features, target_arr)
#xe = ['This is the best car that I have seen so far, so great yeah yeah','Bad service and it really sucks']

predicted = text_clf.predict(x_test)
#print(predicted)
print(np.mean(predicted == y_test))
