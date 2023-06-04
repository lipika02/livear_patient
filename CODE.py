# importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# reading dataset
df = pd.read_excel("C:/Users/lipik/Desktop/rp/indian_liver_patient.xlsx")

# describing dataset
print(df.describe().T)  #Values need to be normalized before fitting. 

# checking null values
print(df.isnull().sum())
#df = df.dropna()

# calculating avg values
print(df['Albumin_and_Globulin_Ratio'].mean())

# filling nan with avg values
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(0.947)
print(df.isnull().sum())

categories = {1:1, 2:0}

df['Dataset']=df['Dataset'].replace(categories)

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'Dataset':'Label'})
print(df.dtypes)

#=========================================================================#
#Understand the data.
# Value 1 = Liver disease and 2 is no disease
sns.countplot(x="Label", data=df)
sns.countplot(x="Label", hue="Gender", data=df) # labels based on genders.

# age range graph
sns.distplot(df['Age'], kde=False)

# liver patients with age.
plt.figure(figsize=(20,10)) 
sns.countplot(x = 'Age', data = df, order = df['Age'].value_counts().index)

sns.scatterplot(x="Label", y="Albumin", data=df)  #Seems no trend between labels 1 and 2
sns.scatterplot(x="Label", y="Albumin_and_Globulin_Ratio", data=df)  #Seems no trend between labels 1 and 2
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio", data=df)  #Seems no trend between labels 1 and 2

#sns.pairplot(df, hue='Gender')

# gives ratio
corr=df.corr()
plt.figure(figsize=(20,12)) 
sns.heatmap(corr,cmap="Blues",linewidths=.5, annot=True)
# Maybe Gender and total protien not big factors influencing the label

#Replace categorical values with numbers
df['Gender'].value_counts()

categories = {"Male":1, "Female":2}
df['Gender'] = df['Gender'].replace(categories)

#Define the dependent variable that needs to be predicted (labels)
Y = df["Label"].values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "Gender"], axis=1) 

# normalize
from keras.utils import normalize
X = normalize(X, axis=1)

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


#############################################################################
#                       MODELS WITHOUT UPSAMPLING
#############################################################################

# MODEL 1: Logistic regression
from sklearn.linear_model import LogisticRegression

model_logistic = LogisticRegression(max_iter=50).fit(X, Y)

prediction_test_LR = model_logistic.predict(X_test)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_LR))
# accuracy = 76.27%



# MODEL 2: SVM
from sklearn.svm import SVC

model_SVM = SVC(kernel='linear')
model_SVM.fit(X_train, y_train)

prediction_test_SVM = model_SVM.predict(X_test)
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_SVM))
# accuracy = 76.27%



#MODEL 3: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators = 25, random_state = 42)

# Train the model on training data
model_RF.fit(X_train, y_train)

#importances = list(model_RF.feature_importances_)
features_list = list(X.columns)
feature_imp = pd.Series(model_RF.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)

#Test prediction on testing data. 
prediction_test_RF = model_RF.predict(X_test)

#ACCURACY METRICS
print("********* METRICS FOR IMBALANCED DATA *********")
#Let us check the accuracy on test data
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_RF))
# accuracy = 72.88%


###########################################################################
# for random forest
# livear patient with and without disease in the training dataset.
(unique, counts) = np.unique(prediction_test_RF, return_counts=True)
print(unique, counts)

#Confusion Matrix: shows how much data is predicted correctly and not.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_test_RF)
print(cm)

#Print individual accuracy values for each class, based on the confusion matrix
print("With Lung disease = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("No disease = ", cm[1,1] / (cm[0,1]+cm[1,1]))
# accuracy (with=79.59%, without=40%)


# ROC GRAPH PLOTTING
from sklearn.metrics import roc_auc_score  #Version 0.23.1 of sklearn

print("ROC_AUC score for imbalanced data is:")
print(roc_auc_score(y_test, prediction_test_RF))
# 57.61%


#https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
from yellowbrick.classifier import ROCAUC

roc_auc=ROCAUC(model_RF)  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()


# SMOTE: Synthetic Minority Oversampling Technique
# ADASYN: Adaptive Synthetic
# SMOTE may not be the best choice all the time. It is one of many things
#that you need to explore. 

from imblearn.over_sampling import SMOTE, ADASYN

X_smote, Y_smote = SMOTE().fit_resample(X, Y)  #Beware, this takes some time based on the dataset size
#X_adasyn, Y_adasyn = ADASYN().fit_resample(X, Y)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, Y_smote, test_size=0.2, random_state=42)

(unique, counts) = np.unique(Y, return_counts=True)
print("Original data: ", unique, counts)
(unique2, counts2) = np.unique(Y_smote, return_counts=True)
print("After SMOTE: ", unique2, counts2)
#(unique3, counts3) = np.unique(Y_adasyn, return_counts=True)
#print("After ADASYN: ", unique3, counts3)

model_SMOTE = RandomForestClassifier(n_estimators = 20, random_state = 42)
model_SMOTE.fit(X_train_smote, y_train_smote)

prediction_test_smote = model_SMOTE.predict(X_test_smote)

print ("Accuracy = ", metrics.accuracy_score(y_test_smote, prediction_test_smote))
# acurracy = 82.63%, 80.23

print(roc_auc_score(y_test_smote, prediction_test_smote))
# 82.39%, 80.21

from yellowbrick.classifier import ROCAUC
roc_auc=ROCAUC(model_SMOTE)
roc_auc.fit(X_train_smote, y_train_smote)
roc_auc.score(X_test_smote, y_test_smote)
roc_auc.show()



############################################################################
#                       HANDLING IMBALANCED DATA
############################################################################

# Up-sample minority class
from sklearn.utils import resample
print(df['Label'].value_counts())

#Separate majority and minority classes
df_majority = df[df['Label'] == 1]
df_minority = df[df['Label'] == 2]

# Upsample minority class and other classes separately
# If not, random samples from combined classes will be duplicated and we run into
#same issue as before, undersampled remians undersampled.
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=416,    # to match average class
                                 random_state=42) # reproducible results
 

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled['Label'].value_counts())  # equally distributed.

# new dataset
Y_upsampled = df_upsampled["Label"].values

#Define the independent variables
X_upsampled = df_upsampled.drop(labels = ["Label", "Gender"], axis=1) 
X_upsampled = normalize(X_upsampled, axis=1)

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled = train_test_split(X_upsampled, Y_upsampled, test_size=0.2, random_state=20)           
                        
#############################################################################                                        
#                       MODELS WITHOUT UPSAMPLING
#############################################################################

# MODEL 1: Logistic regression
from sklearn.linear_model import LogisticRegression

model_logistic = LogisticRegression(max_iter=50).fit(X_upsampled, Y_upsampled)

prediction_test_LR = model_logistic.predict(X_test_upsampled)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test_upsampled, prediction_test_LR))
# accuracy = 64.07%



# MODEL 2: SVM
from sklearn.svm import SVC

model_SVM = SVC(kernel='linear')
model_SVM.fit(X_train_upsampled, y_train_upsampled,)

prediction_test_SVM = model_SVM.predict(X_test_upsampled)
print ("Accuracy = ", metrics.accuracy_score(y_test_upsampled, prediction_test_SVM))
# accuracy = 58.08%
 


# MODEL 3: RANDOM FOREST
# Train again with new upsamples data
model_RF_upsampled = RandomForestClassifier(n_estimators = 20, random_state = 42)

# Train the model on training data
model_RF_upsampled.fit(X_train_upsampled, y_train_upsampled)
prediction_test_RF_upsampled = model_RF_upsampled.predict(X_test_upsampled)

print("********* METRICS FOR BALANCED DATA USING UPSAMPLING *********")

print ("Accuracy = ", metrics.accuracy_score(y_test_upsampled, prediction_test_RF_upsampled))
# ACCURACY = 88.02%

# CONFUSION MATRIX
cm_upsampled = confusion_matrix(y_test_upsampled, prediction_test_RF_upsampled)
print(cm_upsampled)

print("With Lung disease =  = ", cm_upsampled[0,0] / (cm_upsampled[0,0]+cm_upsampled[1,0]))
print("No lung disease = ",  cm_upsampled[1,1] / (cm_upsampled[0,1]+cm_upsampled[1,1]))
# ACCURACY (WITH=90.69%, WITHOUT=85.18)

print("ROC_AUC score for balanced data using upsampling is:")
print(roc_auc_score(y_test_upsampled, prediction_test_RF_upsampled))
# 88.13%

from yellowbrick.classifier import ROCAUC

roc_auc=ROCAUC(model_RF_upsampled)
roc_auc.fit(X_train_upsampled, y_train_upsampled)
roc_auc.score(X_test_upsampled, y_test_upsampled)
roc_auc.show()

############################################################################
#               Generate synthetic data (SMOTE and ADASYN)
############################################################################

# SMOTE: Synthetic Minority Oversampling Technique
# ADASYN: Adaptive Synthetic
# SMOTE may not be the best choice all the time. It is one of many things
#that you need to explore. 

from imblearn.over_sampling import SMOTE, ADASYN

X_smote, Y_smote = SMOTE().fit_resample(X_upsampled, Y_upsampled)  #Beware, this takes some time based on the dataset size
#X_adasyn, Y_adasyn = ADASYN().fit_resample(X, Y)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, Y_smote, test_size=0.2, random_state=42)

(unique, counts) = np.unique(Y, return_counts=True)
print("Original data: ", unique, counts)
(unique2, counts2) = np.unique(Y_smote, return_counts=True)
print("After SMOTE: ", unique2, counts2)
#(unique3, counts3) = np.unique(Y_adasyn, return_counts=True)
#print("After ADASYN: ", unique3, counts3)

model_SMOTE = RandomForestClassifier(n_estimators = 20, random_state = 42)
model_SMOTE.fit(X_train_smote, y_train_smote)

prediction_test_smote = model_SMOTE.predict(X_test_smote)

print ("Accuracy = ", metrics.accuracy_score(y_test_smote, prediction_test_smote))
# accuracy = 86.22%

print(roc_auc_score(y_test_smote, prediction_test_smote))
# 85.65%

from yellowbrick.classifier import ROCAUC
roc_auc=ROCAUC(model_SMOTE)
roc_auc.fit(X_train_smote, y_train_smote)
roc_auc.score(X_test_smote, y_test_smote)
roc_auc.show()


############################################################################
#                             ENSEMBLE METHODS
############################################################################

# BASIC ENSEMBLE METHODS

# AVERAGINH METHOD
from sklearn.metrics import mean_squared_error

pred1 = prediction_test_LR
pred2 = prediction_test_SVM
pred3 = prediction_test_RF_upsampled

pred_final = (pred1+pred2+pred3)/3.0
 
# printing the mean squared error between real value and predicted value
print(mean_squared_error(y_test_upsampled, pred_final))


############################################################################
# MAX VOTING
from sklearn.metrics import log_loss
from sklearn.ensemble import VotingClassifier

model_1 = model_logistic
model_2 = model_SVM
model_3 = model_RF_upsampled

# Making the final model using voting classifier
final_model = VotingClassifier(estimators=[('logr', model_1), ('svm', model_2), ('rf', model_3)], voting='hard')
 
# training all the model on the train dataset
final_model.fit(X_train_upsampled, y_train_upsampled)
 
# predicting the output on the test dataset
pred_final = final_model.predict(X_test_upsampled)
 
# printing log loss between actual and predicted value
print(log_loss(y_test_upsampled, pred_final))

############################################################################
# ADVANCED ENSEMBLE METHODS

# STACKING

from sklearn.metrics import mean_squared_error
 
# importing stacking lib
from vecstack import stacking

model_1 = model_logistic
model_2 = model_SVM
model_3 = model_RF_upsampled
 
# putting all base model objects in one list
all_models = [model_1, model_2, model_3]
 
# computing the stack features
s_train, s_test = stacking(all_models, X_train_upsampled, X_test_upsampled, y_train_upsampled, regression=True, n_folds=4, random_state=None)
 
# initializing the second-level model
final_model = model_2
 
# fitting the second level model with stack features
final_model = final_model.fit(s_train, y_train_upsampled)
 
# predicting the final output using stacking
pred_final = final_model.predict(X_test_upsampled)
 
# printing the mean squared error between real value and predicted value
print(mean_squared_error(y_test_upsampled, pred_final))

###########################################################################################################
# VOTING ENSEMBLE

#load dataset
from sklearn import model_selection
seed = 42
kfold = model_selection.KFold(n_splits=20)

# create different models
estimators = []

model_1 = LogisticRegression(); estimators.append(('logistic',model_1))
model_2 = SVC(); estimators.append(('svm',model_2))
model_3 = RandomForestClassifier(); estimators.append(('rf',model_3))

# create the ensemble model
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X_train_upsampled, y_train_upsampled, cv=kfold)
print(results.mean())
# 62.36%

###############################################################################################################
# TUNE ENSEMBLE
from sklearn.model_selection import cross_val_score, GridSearchCV

lr = LogisticRegression(solver='liblinear', random_state=1)
rf = RandomForestClassifier(max_features=9, n_jobs=4, random_state=1)
sv = SVC()

# create an ensemble of 3 classifiers
vc = VotingClassifier([('clf1',lr), ('clf2',rf), ('clf3',sv)])

cross_val_score(vc,X_train_upsampled, y_train_upsampled).mean()

# define VotingClassifier parameters to search
param = {'voting':['hard', 'soft'], 'weights': [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]}

#find the best set of parameters
grid = GridSearchCV(vc, param)
grid.fit(X_train_upsampled, y_train_upsampled)

# what accuracy is now ?
grid.best_score_
#77.89%







########################################################################################################
#                               k-folds validation
########################################################################################################
from sklearn.model_selection import cross_val_score
# 20 folds
scores = cross_val_score(model_RF_upsampled, X_train_upsampled, y_train_upsampled, scoring='r2', cv=20)

print(scores)

print(np.mean(scores))

from sklearn.model_selection import cross_val_predict
pred = cross_val_predict(model_RF_upsampled, X_test_upsampled, y_test_upsampled)
print(pred)
# 36.17%

scores_test = cross_val_score(model_RF_upsampled, X_test_upsampled, y_test_upsampled, cv=20)
print(scores_test)

print(np.mean(scores_test))
# 71.45%







