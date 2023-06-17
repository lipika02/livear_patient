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

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'Dataset':'Label'})
print(df.dtypes)

categories = {1:1, 2:0}
df['Label'] = df['Label'].replace(categories)

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
#                       MODELS WITH UPSAMPLED DATASET
#############################################################################


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


########################################################################################################
#                               k-folds validation
########################################################################################################
from sklearn.model_selection import cross_val_score
# 10 folds
scores = cross_val_score(model_SMOTE, X_train_smote, y_train_smote, scoring='r2', cv=10)

print(scores)
#70.11%

print(np.mean(scores))
# 30.19%

from sklearn.model_selection import cross_val_predict
pred = cross_val_predict(model_SMOTE, X_test_smote, y_test_smote)
print(pred)

scores_test = cross_val_score(model_SMOTE, X_test_smote, y_test_smote, cv=10)
print(scores_test)

print(np.mean(scores_test))
# 75.51%


#############################################################################################################
#                                       Xgboost MODEL
#############################################################################################################
import xgboost as xgb

# Fitting XGBoost to the training data
my_model = xgb.XGBClassifier()
my_model.fit(X_train, y_train)
 
# Predicting the Test set results
y_pred = my_model.predict(X_test)
print(y_pred)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print ("Accuracy = ", my_model.score(X_test, y_test))
# 71.18%

#############################################################################################################
#                                       LIGHTGBM MODEL
#############################################################################################################
import lightgbm as lgb
from lightgbm import LGBMClassifier

model = LGBMClassifier()
model.fit(X_train, y_train)
 
# Predicting the Target variable
pred = model.predict(X_test)
print(pred)
accuracy = model.score(X_test, y_test)
print(accuracy)
# 69.49%
























