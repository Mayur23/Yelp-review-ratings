# Yelp-review-ratings



Data analysis and vizualization :- 


FacetGrid, a grid of 5 histograms, from the Seaborn library is created to explore the distribution of the length of text across the 5 stars. Here, we determine that the distribution of text length across the reviews are very similar for all the stars but the amount of text reviews are much higher for the 4-star and 5-star movies compared to the lower rated movies. This may prove to be an issue as we go along the proccess. 

Box plot created shows that important insights can not be determined from Text length because of the large number of outliers. 

A Countplot of the number of occurances for each type of star rating is created to reflect the results shown in the Facetgrid plot (4 and 5 star ratings have more number of reviews)

Mean Values of numerical columns are determined and represented in a new dataframe. From this dataframe, Correlation between the columns Text Length, Funny, Useful, and Cool is plotted on a Heat map. 


NLP classification :-


A CountVecorizer object is created, pass in the text column created earlier.

Split up the data into training and testing data, train the model based on on Multinomial Naive Bayes classifier. 

Use the predict method off of Naiver Bayes classifier to predict labels from the test data. 

A confusion matrix and classification report is created using the predictions from the above step and and the y_test data.

TF-IDF statistic is included using a pipeline with CountVectorizer, TF-IDF Transformer, and Multinomial classifier. repeat the splitting and training step.

