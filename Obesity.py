import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

pd.pandas.set_option("display.max_columns", None)
data_set = pd.read_csv("C:/Users/Admin/Documents/UCI repository/Obesity Folder/ObesityPredictionParameters.csv")
print(data_set.head(20))
print(data_set.info())
'''There are nine categorical features. Of these nine, we need to encode some with a grading system depending on 
the entries to retain the order. 
There are no null values which means we can skip to encoding the categoricals.

Gender,FamilyObesityHistory,FreqHighCalFood,SMOKE,CalorieConsMonitoring
'''
factorizables = ['Gender', 'FamilyObesityHistory', 'FreqHighCalFood', 'SMOKE', 'CalorieConsMonitoring']
temp_df = [[]]
temp_df = pd.DataFrame(temp_df)
for feature in factorizables:
    temp_series = data_set[feature].factorize()
    temp_series = pd.Series(temp_series[0], name=feature)
    temp_df = pd.concat([temp_df, temp_series], axis=1)

data_set = data_set.drop(factorizables, axis=1)
data_set = pd.concat([data_set, temp_df], axis=1)

#ENCODING SNACKS
print(data_set['Snacks'].unique())
#We can see there are four adjectives used in the snacks feature.
snacks_dict = {'Sometimes': 1, 'Frequently': 2, 'Always': 3, 'no': 0}
data_set.replace(snacks_dict, inplace=True)

#ENCODING MODE OF TRANSPORT
print(data_set['ModeOfTransport'].unique())
trans_dict = {'Public_Transportation': 3,  'Walking': 0, 'Automobile': 4, 'Motorbike': 2, 'Bike': 1}
data_set.replace(trans_dict, inplace=True)

#ENCODING OBESITY LEVEL (THE TARGET)
print(data_set['Obesity Level'].unique())
obesity_dict = {'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I': 4,
                 'Insufficient_Weight': 0, 'Obesity_Type_II': 5,  'Obesity_Type_III': 6}
data_set.replace(obesity_dict, inplace=True)


#Now that all the data is encoded. We can move on to visualising the data
sns.heatmap(data_set.corr(), annot=True, cmap='plasma')
plt.show()
''' We can see from the heat-map that there is generally weak correlation between obesity levels and the independent 
 features save for two; family history and weight. There is a strong Pearson correlation between weight and obesity.
 In terms of family history, there is a decent negative correlation between the family obesity history as can be 
 expected. This is because of factorisation. It should not affect the model.
'''

y = data_set.pop('Obesity Level')
X_train, X_test, y_train, y_test = train_test_split(data_set, y, test_size=0.3, random_state=0)

#LOGISTIC REGRESSION
log_est = LogisticRegression()
log_est.fit(X_train, y_train)
first_pred = log_est.predict(X_test)
result = classification_report(y_test, first_pred)
print(result)


#SUPPORT VECTOR MACHINE CLASSIFIER
svm_est = SVC()
svm_est.fit(X_train, y_train)
second_pred = svm_est.predict(X_test)
result_2 = classification_report(y_test, second_pred)
print(result_2)


#DECISION TREE CLASSIFER
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
third_pred = decision_tree.predict(X_test)
result_3 = classification_report(y_test, third_pred)
print(result_3)

#GRADIENT BOOSTING CLASSIFIER
grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train, y_train)
fourth_pred = grad_boost.predict(X_test)
result_4 = classification_report(y_test, fourth_pred)
print(result_4)

#XG-BOOST CLASSIFIER
xg_boost = XGBClassifier()
xg_boost.fit(X_train, y_train)
fifth_pred = xg_boost.predict(X_test)
result_5 = classification_report(y_test, fifth_pred)
print(result_5)

#We could have added a NN or hyperparameter tunning for estimators like XG-boost and the ensemble techniques
# but 99% precision and 98% recall is good enough so we can save the trained XG-Boost model

pickle.dump(grad_boost, open('obesity_grad_model.bst', 'wb'))


