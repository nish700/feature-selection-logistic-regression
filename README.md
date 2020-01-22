### Project Overview

 **Spam Vs Non-Spam**

In the project , we classify Spam mails in the given dataset.  This is achieved using Logistic Regression model . Initiallly a baseline model is created containing all the features. Using correlation matrix , the highly correlated features are removed and then a second model with lesser number of features is modelled. Similarly , Chi square and Annova model are used to furthure refine the features to achieve better accuracy. Finally PCA is applied to the dataset for feature extraction for better test accuracy. 

**About the dataset:**

•	Number of Instances: 4601 (1813 Spam = 39.4%)

•	Number of Attributes: 58 (57 continuous, 1 nominal class label)

•	Attribute Information:

    o	The last column of 'spambase.data' denotes whether the e-mail was considered spam (1) or not (0)

    o	48 attributes are continuous real [0,100] numbers of type word freq WORD i.e. percentage of words in the e-mail that match WORD

    o	6 attributes are continuous real [0,100] numbers of type char freq CHAR i.e. percentage of characters in the e-mail that match CHAR

    o	1 attribute is continuous real [1,…] numbers of type capital run length average i.e. average length of uninterrupted sequences of capital letters

    o	1 attribute is continuous integer [1,…] numbers of type capital run length longest i.e. length of longest uninterrupted sequence of capital letters

    o	1 attribute is continuous integer [1,…] numbers of type capital run length total i.e. sum of length of uninterrupted sequences of capital letters in the email

    o	1 attribute is nominal {0,1} class of type spam i.e denotes whether the e-mail was considered spam (1) or not (0)

•	Missing Attribute Values: None

•	Class Distribution: Spam 1813 (39.4%) Non-Spam 2788 (60.6%)


**Following concepts have been implemented in this project:**

•	Logistic Regression

•	Correlation Matrix

•	Classification Report

•	Confusion Matrix

•	Chi square test

•	Annova model

•	PCA model

•	SelectKBest





