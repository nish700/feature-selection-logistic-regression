# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 

# load the data
df = pd.read_csv(path , header=None)

# get overview of the data
print(df.shape)
print(df.info())
print(df.describe())


# check the frequency distribution of spam and non-spam
freq_dist = df.iloc[:,57].value_counts(normalize=True)
print("Frequency distribution of span and non-spam is:", freq_dist)

# split the data into features and labels
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# # splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.2)

# check the shape of train and test set
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# initialize the Logistic Regression Model
lr_model = LogisticRegression()
# fit the model to train data
lr_model.fit(X_train, y_train)
# predict using the model
y_pred = lr_model.predict(X_test)
#calculate the accuracy
accuracy = lr_model.score(X_test, y_test)

print('Accuracy of the baseline LR model is:', accuracy)

# print classification report
class_report = classification_report(y_test, y_pred)
print('Classification report of the baseline model is:', class_report)

# print the Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix for the classification is:', conf_matrix)

print('******************'*20)

# remove correlated features and make the predictions
# make a copy of the dataframe , so that a change in new dataframe does not in any way impact the #original dataframe
df1 = df.copy()

# find the correlation of df1 (removing the label from the df1 for the same)
corr_mat = df1.iloc[:,:-1].corr()

# plot the heat map for the correlation
sns.heatmap(corr_mat, cmap='YlGnBu')
plt.show()

# removing the highly correlated features(ones having correlation higher than 0.75) to satisfy the LR 
# assumption of multicollinearity
# extract the upper triangle of the correlation matrix (it's symetric), excluding the diagonal, For 
# this we are going to use np.tril, cast this as a boolean, and get the opposite of it using the ~ 
# operator. 
corr_triu = corr_mat.where(~np.tril(np.ones(corr_mat.shape)).astype(np.bool))

# using list comprehension to filter the columns to drop
columns_to_drop = [column for column in corr_triu.columns if any(corr_triu[column] > 0.75)]
print("Features with high correlation: ",columns_to_drop)

# drop the columns with high correlation
df_new = df1.drop(df.columns[columns_to_drop], axis=1)
print("shape of the modified dataframe is:", df_new.shape)

# split the new dataset into train and test set
X_new = df_new.iloc[:,:-1]
y_new = df_new.iloc[:,-1]

X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(X_new, y_new, random_state=0, test_size=0.2)

# initialise the LR model, fit and predict
lr_model_new = LogisticRegression()
lr_model_new.fit(X_n_train, y_n_train)
y_n_pred = lr_model_new.predict(X_n_test)

# calculate the accuracy
accuracy_new = lr_model_new.score(X_n_test, y_n_test)
print('Accuracy score of modified LR model is:', accuracy_new)

# Calculate the classification report
class_report_new = classification_report(y_n_test,y_n_pred)
print('Classification report for the modified LR model is:', class_report_new)

# confusion matrix for the modified model
conf_matrix_new = confusion_matrix(y_n_test, y_n_pred)
print('Confusion matrix for the modified LR model is :', conf_matrix_new)

print('******************'*20)


#splitting the data into train and test set
X_t = df_new.iloc[:,:-1]
y_t = df_new.iloc[:,-1]
X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_new, y_new, random_state=0, test_size=0.2)

# initialise the selecdKbest model using chi2 test

optimum_features = 0
y_pred_t_final = []

# function to generalise the SelectKbest model for feature selection
def optimum_features(func, n_features):
    accuracy_transformed = 0
    # looping to select best combination of features    
    for n_feature in range(1,n_features):
        # initialise the selecdKbest model
        test = SelectKBest(score_func = func, k=n_feature)

        # fit and tranform the new train data
        X_train_transformed = test.fit_transform(X_t_train, y_t_train)
        X_test_transformed = test.transform(X_t_test)

         # initialise the LR model and fit the data to it
        lr_model_trans = LogisticRegression()
        lr_model_trans.fit(X_train_transformed, y_t_train)
        y_pred_trans = lr_model_trans.predict(X_test_transformed)

        # calculate the accuracy of the model
        acc_trans = lr_model_trans.score(X_test_transformed, y_t_test)
        #save the best accuracy and optimum features
        if accuracy_transformed < acc_trans:
            accuracy_transformed = acc_trans
            optimum_features = n_feature
            y_pred_t_final = y_pred_trans
    
    return [accuracy_transformed,optimum_features,y_pred_t_final]


# optimum number of feature from chi2 test
accuracy_chi2_test , optimum_features_chi2 ,y_pred_t_final_chi2 = optimum_features(func= chi2 ,n_features = X_t_train.shape[1])

print('Best Accuracy of the model after chi2 transformation is {0} with {1} no. of features'.format(accuracy_chi2_test,optimum_features_chi2))

# classification report of the tranformed model
class_report_transformed_chi2 = classification_report(y_t_test, y_pred_t_final_chi2)
print('Classification report after chi2 transformation:', class_report_transformed_chi2)

# confusion matrix of the transformed model
conf_matrix_transformed_chi2 = confusion_matrix(y_t_test, y_pred_t_final_chi2)
print('Confusion matrix for the transformed model:', conf_matrix_transformed_chi2)

print('******************'*20)

# Find the optimum number of features using Anova and fit the logistic model on train data.
# calling the generalised function optimum_features and getting the best features

accuracy_f_classif , optimum_features_f_classif ,y_pred_t_final_f_classif = optimum_features(func = f_classif , n_features = X_t_train.shape[1])

# print accuracy
print("Annova selection has highest accuracy of {0} with {1} no. of features".format(accuracy_f_classif,optimum_features_f_classif))

# print classification report
class_report_transformed_f_classif = classification_report(y_t_test, y_pred_t_final_f_classif)
print('Classification report for Annova feature selection is:', class_report_transformed_f_classif)

# print the confusion matrix
conf_matrix_transformed_f_classif = confusion_matrix(y_t_test, y_pred_t_final_f_classif)
print('Confusion matrix for Annova selection is:', conf_matrix_transformed_f_classif)

print('******************'*20)

# Applying PCA transformation on the data
n_components = X_t_train.shape[1]
best_accuracy_pca = 0
best_n_features_pca = 0 
y_best_pred_pca = []

for n_component in range(1,n_components):
    # initialise pca for n_feature
    pca = PCA(n_component, random_state=0)
    
    #fit and transform the train and test data
    X_train_pca = pca.fit_transform(X_t_train)
    X_test_pca = pca.transform(X_t_test)

    # initialise the LR model
    model_LR = LogisticRegression(random_state=0)
    # fit the model to train data
    model_LR.fit(X_train_pca, y_t_train)
    # calculate the score 
    y_pred_pca = model_LR.predict(X_test_pca)

    # calculate accuracy
    accuracy_pca = model_LR.score(X_test_pca, y_t_test)

    if accuracy_pca > best_accuracy_pca :
        best_accuracy_pca = accuracy_pca
        best_n_features_pca = n_component
        y_best_pred_pca = y_pred_pca 

# print pca accuracy
print("Accuracy score of PCA model is {0} with {1} no. of features".format(best_accuracy_pca, best_n_features_pca))

# print confusion matrix
conf_matrix_pca = confusion_matrix(y_t_test, y_best_pred_pca)
print('Confusion matrix employing PCA transformation is:', conf_matrix_pca)



