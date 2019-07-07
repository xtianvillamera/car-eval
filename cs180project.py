#!usr/bin/env python3

import os
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn import tree

def GetInput():
	print('\n')
	print('Hello! You want to sell your car but first, let us evaluate it.\n')
	buy_price = input('What its buying price? (vhigh,high,med,low)\n')
	maint_price = input('What is the price of its maintenance? (vhigh,high,med,low)\n')
	door_num = input('How many doors does your car have? (2,3,4,5more)\n')
	person_cap = input('What its maximum passenger\'s capacity? (2,4,more)\n')
	lug_boot = input('What is the size of the luggage we can put on it? (small,med,big)\n')
	safety = input('What is the overall safeness of the car? (low,med,high)\n')		
	comment = input('Your comment about the car: \n')
	return buy_price,maint_price,door_num,person_cap,lug_boot,safety,comment

# I
def Convert(attr,val):
	if attr == 0 or attr == 1 or attr == 6:
		if val == 'vhigh' or val == 'unacc':
			return 1
		elif val == 'high' or val == 'acc':
			return 2
		elif val == 'med' or val == 'good':
			return 3
		elif val == 'low' or val == 'vgood':
			return 4
	if attr == 2 or attr == 3:
		if val == '2':
			return 2
		elif val == '3':
			return 3
		elif val == '4':
			return 4
		elif val == 'more' or val == '5more':
			return 5
		else:
			return 66	
	if attr == 4 or attr == 5:
		if val == 'small' or val == 'low':
			return 1
		elif val == 'med':
			return 2
		elif val == 'big' or val == 'high':
			return 3

def EvaluationTraining():			
	file = open('careval_dataset.txt','r')
	#file = open('sample.txt','r')
	with file as f:
		line = f.readlines()

	content = [x.strip() for x in line] 
	num_lines = len(content)

	feature = []
	predict = []
	word_feature = []
	for attr in content:
		attributes = attr.split(',')
		word_feature.append(attributes)
		car = []
		for i,a in enumerate(attributes):
			if i != 6:
				car_attr = Convert(i,a)
				car.append(car_attr)
			elif i == 6:
				p = Convert(i,a)
				predict.append(p)

		feature.append(car)

	word_feature_arr = np.array(word_feature)	
	feature_array = np.array(feature, 'float64')
	predict_array = np.array(predict, 'float64')
	clf = tree.DecisionTreeClassifier()
	clf.fit(feature_array,predict_array)

	file.close()	
	return clf

def SentimentAnalysis():
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

	#print(features_no_sw)
	#print(stringh)


	target_arr = np.array(target)		
	x_train, x_test, y_train, y_test = train_test_split(features, target_arr, test_size = 0.2, random_state=42)

	text_clf = Pipeline([('vect',CountVectorizer()),('tf', TfidfTransformer()),('clf',MultinomialNB())])
	#text_clf = Pipeline([('vect',CountVectorizer()),('tf', TfidfTransformer()),('clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,max_iter=5, tol=None))])
	text_clf.fit(features, target_arr)
	#xe = ['This is the best car that I have seen so far, so great yeah yeah','Bad service and it really sucks']

	#predicted = text_clf.predict(x_test)
	#print(predicted)
	#print(np.mean(predicted == y_test))	
	return text_clf

def Results(evalu,sentiment):
	if sentiment == 0:
		if evalu == 1:
			return "Your car is not in good condition and customers will not like it. I'm sorry to say but your car isn't suited for selling."
		elif evalu == 2:
			return "Your car has some issues but it can still be sold for a very cheap price."
		elif evalu == 3:
			return "Your car is a little bit rusty but can still do wonders on the road. It can be sold for a reasonable price."
		elif evalu == 4:
			return "You think that your car is in poor condition but some customers might like it and definitely will pay for it."
	elif sentiment == 1:
		if evalu == 1:
			return "You might think that your car is still good but the customer might not like it. Sorry but it looks like no one would buy it."
		elif evalu == 2:
			return "Your car is in good shape and works fine. It can be sold for a reasonable price."
		elif evalu == 3:
			return "You have a well maintained car. Buyers would surely be satisfied with it."
		elif evalu == 4:		
			return "You have an amazing car!! It's still in a very good condition. Customers will definitely like it and it is certain that it can be sold at a valuable price."
cont = 'yes'

while cont == 'yes':
	inputs = GetInput()
	evaluation = inputs[:6]
	EvalClassifier = EvaluationTraining()
	car_eval = []
	for i,e in enumerate(evaluation):
		car_eval.append(Convert(i,e))
	car_eval_arr = np.array(car_eval)	
	#print(car_eval_arr)
	comment = inputs[6]
	SentimentClassifier = SentimentAnalysis()
	e_class = EvalClassifier.predict([car_eval_arr])
	s_class = SentimentClassifier.predict([comment])
	print('\n')
	print(e_class,s_class)
	print('Here is our evaluation on your car: ')
	print(Results(int(e_class[0]),int(s_class[0])))
	print('\n')
	cont = input('Would you like to evaluate another car? (yes,no)\n')
