
# coding: utf-8

# In[1]:


#!/usr/bin/python
import pprint
import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import mean
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
poi = ['poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']
my_features_list = poi + email_features + financial_features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# All data points in the data set
enron_data = pd.DataFrame.from_dict(data_dict, orient = 'index')
enron_data.head()
print "There are a total of {} people in the dataset." .format(len(enron_data.index)) 
print "there are  {} POI and {} Non-POI." .format(enron_data['poi'].value_counts()[True], 
                                                 enron_data['poi'].value_counts()[False])
print "Total number of email plus financial features are {}. 'poi' column is the label." .format(len(enron_data.columns)-1)
enron_data.describe().transpose()


### Task 2: Remove outliers
#plot the outliers
def PlotOutlier(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'black'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

# Visualise outliers
print(PlotOutlier(data_dict, 'total_payments', 'total_stock_value'))
print(PlotOutlier(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(PlotOutlier(data_dict, 'bonus', 'salary'))
#Remove outlier TOTAL
data_dict.pop( 'TOTAL', 0 )


#remove other outliers
def remove_outlier(dict_object, keys):
    for key in keys:
        dict_object.pop(key, 0)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outlier(data_dict, outliers)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0
new_features_list = my_features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio']
print (new_features_list)


# In[2]:

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Selecting the best features: 
#Removes all features whose variance is below .8 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)
#print len(features)
# k highest scoring features
from sklearn.feature_selection import f_classif
k = 6
selector = SelectKBest(f_classif, k= 6)
selector.fit_transform(features, labels)
print("KBest features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
pprint.pprint (sorted_scores)
features_list = poi + list(map(lambda x: x[0], sorted_scores))[0:k]
print("final features list :")
pprint.pprint (features_list)

# dataset without new features for testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Extract new features for testing
data = featureFormat(my_dataset, features_list +                      ['msg_to_poi_ratio', 'msg_from_poi_ratio'],                      sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)


# # classifiers- Naive Bayes

# In[3]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

# Naive Bayes classifier
pipe_nb = Pipeline([
  ('scaler', StandardScaler()),
  ('pca', PCA()),
  ('clf', GaussianNB())
])

n_components = [1, 2, 3, 4, 5, 6]

estimator = GridSearchCV(
    pipe_nb, 
    dict(
        pca__n_components = n_components
    ),
    scoring = 'f1',
    cv = StratifiedKFold(9, True, 42)
)
estimator.fit(features, labels)
#estimator.fit(new_features, new_labels)

estimator.best_params_


# In[4]:

pipe_nb.set_params(pca__n_components = 2)

#test classifiers
from tester import dump_classifier_and_data, test_classifier

test_classifier(pipe_nb, my_dataset, features_list, folds = 1000)


# # Logistic Regression

# In[5]:

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight="balanced"))
])

estimator = GridSearchCV(
    pipe_lr,
    dict(
        lr__C = [100, 500, 1000, 2000, 3000, 5000, 5500, 6000],
    ),
    scoring = 'f1',
    cv = StratifiedKFold(10, True, 42)
)

estimator.fit(features, labels)
#estimator.fit(new_features, new_labels)

estimator.best_params_


# In[6]:

estimator.best_estimator_.named_steps['lr'].coef_


# In[7]:

pipe_lr.set_params(
    lr__C = 100
)

test_classifier(pipe_lr, my_dataset, features_list, folds = 1000)


# # svm

# In[8]:

pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(class_weight="balanced"))
])

estimator = GridSearchCV(
    pipe_svc,
    dict(
        svc__C = [2000, 5000, 10000],
        svc__gamma = [0.0005, 0.001, 0.005]
    ),
    scoring = 'f1',
    cv = StratifiedKFold(10, True, 42)
)
estimator.fit(features, labels)
#estimator.fit(new_features, new_labels)

estimator.best_params_


# In[9]:

pipe_svc.set_params(svc__C = 5000, svc__gamma = 0.005)
test_classifier(pipe_svc, my_dataset, features_list, folds = 1000)


# # Decision Trees

# In[10]:

pipe_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeClassifier(class_weight = "balanced"))
])

estimator = GridSearchCV(
    pipe_dt,
    dict(
        dt__max_depth = [1, 2, 3, 4, 5, 6]
    ),
    scoring = 'f1',
    cv = StratifiedKFold(10, True, 42)
)
#estimator.fit(features, labels)
estimator.fit(new_features, new_labels)

estimator.best_params_


# In[11]:

pipe_dt.set_params(dt__max_depth = 4)

test_classifier(pipe_dt, my_dataset, features_list, folds = 1000)


# In[12]:

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import naive_bayes
clf = pipe_nb
clf.fit(features, labels)
pipe_nb.set_params(pca__n_components = 2)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



