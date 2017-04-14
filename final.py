data = []
train_y = []

import glob
import os

news_list = glob.glob("news/*")

total_train = 0
data = []



for h in range(0,len(news_list)-1):

    files = glob.glob("news/news_"+str(h+1)+"/*")
    # print (len(files))
    total_train += len(files)
    
    for file in files:
    	
    	train_y.append(h+1)
    	with open(file, 'r') as myfile:
    		
        	data.append(myfile.read())








import re,nltk
from nltk.corpus import stopwords

def review_to_words( raw_review ):
# 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))







clean_train_reviews = []


# print (type(data[4]))
for i in range (0,total_train):
    clean_train_reviews.append(review_to_words(data[i]))



print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
print vectorizer
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# # Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()
# print vocab


import numpy as np

# # Sum up the counts of each vocabulary word
# dist = np.sum(train_data_features, axis=0)

# # For each, print the vocabulary word and the number of times it 
# # appears in the training set
# for tag, count in zip(vocab, dist):
#     print count, tag


print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run

forest = forest.fit( train_data_features ,train_y)







test = []
test_list = glob.glob("news/test_true/*")


paths = os.path.abspath("news/*")
news_len = len(glob.glob(paths))


for my_test in test_list:
    with open(my_test, 'r') as myfile:
        test.append(myfile.read())

clean_test_reviews = []
for i in range(0,len(test_list)):
	clean_test_review = review_to_words( test[i] )

	
	clean_test_reviews.append( clean_test_review )

# print (clean_test_reviews)


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
result_prob = forest.predict_proba(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
import pandas as pd

threshold = 0.3

def filter():
	
	for i in range(0,len(test_list)):
		bl = 0
		for j in range(0,news_len-2):
			
			if result_prob[i][j] >= threshold:
				bl = 1
				break
		
		if bl==0:
			result[i] = 0

			
filter()
output = pd.DataFrame( {"news":np.asarray(test), "class":result} ).set_index('news')

print output



from matplotlib import pyplot as plt

lst = []

for i in range(0,len(result_prob[0])):
    lst.append(i+1)

for i in range(0,len(test)):
    plt.figure(i)
    plt.bar(lst,result_prob[i], align='center', alpha=0.5)
    # plt.scatter(lst,result_prob[i])
    plt.ylabel('Likelihood')
    plt.xlabel('News Classes')
    axes = plt.gca()
    axes.set_ylim([0,1.0])
    plt.axhline(y=0.3, color='r', linestyle='-')
    plt.title(i)
    plt.show()



# gui

# from Tkinter import *
# import Tkinter


# top = Tk()
# frame1 = Tkinter.Frame(top, width=1000, height=1000, background="bisque")

# Lb1 = Listbox(top)

# for i in range(0,len(test)):
#     Lb1.insert(i,test[i])

# Lb1.pack()
# frame1.pack(fill=None, expand=False)
# top.mainloop()


# import Tkinter as tk
# from Tkinter import *

# root = tk.Tk()
# scrollbar = tk.Scrollbar(root, orient="vertical")
# Lb1 = tk.Listbox(root, width=50, height=20, yscrollcommand=scrollbar.set)
# scrollbar.config(command=Lb1.yview)

# scrollbar.pack(side="right", fill="y")
# for i in range(0,len(test)):
#     Lb1.insert(i,test[i])
# Lb1.pack(side="left",fill="both", expand=True)


# root.mainloop()

