#!usr/bin/env python3

import numpy as np
from sklearn import tree

"""
Class Values: 

unacc, acc, good, vgood 
1		2	3		4
Attributes: 

buying: vhigh, high, med, low.
		4		3	 2		1	 
maint: vhigh, high, med, low.
		4		3	2		1 
doors: 2, 3, 4, 5more.
		2	3	4	5 
persons: 2, 4, more.
		2	4	5 
lug_boot: small, med, big.
			1	  2		3 
safety: low, med, high.
		1		2	3	 
"""

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
clf.fit(feature_array[:1700],predict_array[:1700])

print(word_feature_arr[1717:1718])
print(clf.predict(feature_array[1717:1718]))


file.close()	