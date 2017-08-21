import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

yelp = pd.read_csv('yelp.csv')

yelp.head()
yelp.info()
yelp.describe()

yelp['text length'] = yelp['text'].apply(len)

sns.setstyle('white')

//FacetGrid
fg = sns.FacetGrid(yelp,col = 'stars')
fg.map(plt.hist, 'text length)

//Boxplot
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')

//Checking occurances of each type of star
sns.countplot(x='stars',data=yelp,palette='rainbow')

stars = yelp.groupby('stars').mean()

stars.corr()

//heatmap from the above dataframe
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

////////////////////////////////
///////NLP classification///////
////////////////////////////////

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(X)

//Train,Test and split data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

//Predicting results
predictions = nb.predict(X_test)

//Creating confusion matrix and classification report using the predictions
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

//Pipeplines to process text
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()),  
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)

//Predict and Evaluate using the pipeline

predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




