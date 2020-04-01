#!/usr/bin/python

import sys
from final_project import tester
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif
from final_project.miscFunction import testBestParamsDecisionTree
from final_project.miscFunction import testBestParamsKNeighbors
from final_project.miscFunction import getBestFeaturesResult
import matplotlib.pyplot as plt
#from final_project.tester import test_classifier

sys.path.append("../tools/")

#from feature_format import featureFormat, targetFeatureSplit ''' old Version Py 2 '''
#from tester import dump_classifier_and_data ''' old Version Py 2 '''
from tools.feature_format import featureFormat, targetFeatureSplit
from final_project.tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'salary', 'bonus','frac_from_poi_to_this_person_to_total', 'frac_from_this_person_to_poi_to_total'] # You will need to use more features
features_list_all = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person', 'frac_from_poi_to_this_person_to_total', 'frac_from_this_person_to_poi_to_total']
features_list = ['poi']

### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "r") as data_file: ''' old Version Py2'''
#Python 3 adaption pickle dont like '\r\n' so remove them and add a simple '\n'
outsize = 0
content = ''  

with open("final_project_dataset.pkl", 'rb') as data_file:
    content = data_file.read()

with open("final_project_dataset_new.pkl", 'wb') as output:
    for line in content.splitlines():
        line = line.decode() + '\n'
        line = line.encode()
        outsize += len(line) + 1
        output.write(line)
        
enronDataFrame = pd.DataFrame.from_dict(pd.read_pickle('final_project_dataset_new.pkl'), orient='index')
enronDataFrame = enronDataFrame.replace('NaN', np.nan)

### Task 2: Remove outliers
# show dataset Info
print(enronDataFrame.info())
# show how much lines are inside
print('Original length:', len(enronDataFrame))
# here the numbers of POI and non POI
print(enronDataFrame['poi'].value_counts())
# replace all nan with 0
enronDataFrame = enronDataFrame.replace(np.nan, 0)

errorsPayment = (enronDataFrame[enronDataFrame[['salary',
                                                'bonus',
                                                'long_term_incentive',
                                                'deferred_income',
                                                'deferral_payments',
                                                'loan_advances',
                                                'other',
                                                'expenses',                
                                                'director_fees']].sum(axis='columns') != enronDataFrame['total_payments']])

errorsStock = (enronDataFrame[enronDataFrame[['exercised_stock_options',
                                              'restricted_stock',
                                              'restricted_stock_deferred']].sum(axis='columns') != enronDataFrame['total_stock_value']])

print('Payment errors:',errorsPayment.index)
print('Stock errors:',errorsStock.index)
# drop outlier / unwanted columns
del enronDataFrame['email_address']
enronDataFrame.drop(axis=0, labels=['TOTAL','THE TRAVEL AGENCY IN THE PARK','BELFER ROBERT', 'BHATNAGAR SANJAY'], inplace=True)
#check general info after removing Outier
print('*** Data after removed Outlier ***')
print(enronDataFrame.info())
# show how much lines are inside
print('Original length:', len(enronDataFrame))
# here the numbers of POI and non POI
print(enronDataFrame['poi'].value_counts())

# first inital test
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
# convert back to origin data dict
data_dict = enronDataFrame.to_dict(orient='index')

### Store to my_dataset for easy export below.
my_dataset = data_dict

clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

clf = KNeighborsClassifier()
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

clf = DecisionTreeClassifier()
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

### Task 3: Create new feature(s)
# new Feature from eMail data
enronDataFrame['frac_from_poi_to_this_person_to_total'] =  enronDataFrame['from_poi_to_this_person'] / enronDataFrame['to_messages']
enronDataFrame['frac_from_this_person_to_poi_to_total'] =  enronDataFrame['from_this_person_to_poi'] / enronDataFrame['from_messages']
# new Feature from finance data
enronDataFrame['frac_bonus_to_total_payments'] =  enronDataFrame['bonus'] / enronDataFrame['total_payments']
enronDataFrame['frac_salary_to_total_payments'] =  enronDataFrame['salary'] / enronDataFrame['total_payments']
#again replace NaN with 0
enronDataFrame = enronDataFrame.replace(np.nan, 0)

# visualize the new features eMails
plt.scatter(x=enronDataFrame[enronDataFrame['poi']==True]['frac_from_poi_to_this_person_to_total'], y = enronDataFrame[enronDataFrame['poi']==True]['frac_from_this_person_to_poi_to_total'], color = 'red')
plt.scatter(x=enronDataFrame[enronDataFrame['poi']==False]['frac_from_poi_to_this_person_to_total'], y = enronDataFrame[enronDataFrame['poi']==False]['frac_from_this_person_to_poi_to_total'], color = 'blue')
plt.xlabel('from POI / from messages');
plt.ylabel('to POI / to messages');
plt.title('eMail ratios'); 
plt.legend(['POI', 'non POI'])
plt.show()
# visualize the new features finance
plt.scatter(x=enronDataFrame[enronDataFrame['poi']==True]['frac_bonus_to_total_payments'], y = enronDataFrame[enronDataFrame['poi']==True]['frac_salary_to_total_payments'], color = 'red')
plt.scatter(x=enronDataFrame[enronDataFrame['poi']==False]['frac_bonus_to_total_payments'], y = enronDataFrame[enronDataFrame['poi']==False]['frac_salary_to_total_payments'], color = 'blue')
plt.xlabel('bonus / total payments');
plt.ylabel('salary / total payments');
plt.title('bonus and salary ratios'); 
plt.legend(['POI', 'non POI'])
plt.show()
#add the new Features
features_list.append('frac_from_poi_to_this_person_to_total')
features_list.append('frac_from_this_person_to_poi_to_total')
features_list.append('frac_bonus_to_total_payments')
features_list.append('frac_salary_to_total_payments')

#convert back to data dict
my_dataset = enronDataFrame.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

fScores, pValues = f_classif(features, labels)

scoreList = zip(fScores, features_list[1:])
scoreList = sorted(scoreList, key= lambda x:x[0], reverse=True)

print('| Feature | F-Score |')
print('| --- | --- |')
for i in range(len(fScores)):
    print('| {} | {:.4f} |'.format(scoreList[i][1], scoreList[i][0]))
    
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

subfeatures_list = ['shared_receipt_with_poi','restricted_stock', 'total_payments', 'long_term_incentive', 'deferred_income', 'exercised_stock_options', 'total_stock_value', 'bonus', 'frac_bonus_to_total_payments', 'salary', 'frac_from_this_person_to_poi_to_total']
### INPUT
#getBestFeaturesResult (['DecisionTree', 'KNeighbors', 'GaussianNB'], subfeatures_list, my_dataset, score = 'F1')
### OUTPUT best F1 - 11 Feature
#| algorithms | recall | precision | accuracy | F1 | features | runtime | 
#| --- | --- | --- | --- | --- | --- | --- |
#| DecisionTree | 0.4245 | 0.552665799739922 | 0.8319090909090909 | 0.48006785411365566 | ['shared_receipt_with_poi', 'long_term_incentive', 'bonus', 'frac_from_this_person_to_poi_to_total'] | 2:10:19.166714 |
#| KNeighbors | 0.347 | 0.7343915343915344 | 0.8802307692307693 | 0.47130730050933783 | ['shared_receipt_with_poi', 'exercised_stock_options', 'bonus', 'frac_bonus_to_total_payments', 'frac_from_this_person_to_poi_to_total'] | 4:01:46.704527 |
#| GaussianNB | 0.3955 | 0.5566502463054187 | 0.8686428571428572 | 0.46243788365974864 | ['deferred_income', 'exercised_stock_options', 'total_stock_value', 'frac_bonus_to_total_payments', 'salary'] | 2:36:48.203015 |

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Params for DecisionTree
params = {'criterion' : ['gini','entropy'], 
          'splitter' : ['best', 'random'], 
          'max_depth': [None,2,3,4,8,16,32],
          'min_samples_split' : [2,3,4,5,6,7,8,9,10],
          'min_samples_leaf' : [1,2,3,4,5,6],
          'max_features' :[None, 'auto', 'sqrt', 'log2']}

features_list = ['poi', 'shared_receipt_with_poi', 'long_term_incentive', 'bonus', 'frac_from_this_person_to_poi_to_total'] 
### INPUT
#testBestParamsDecisionTree(features_list, my_dataset, params, score='F1')
### OUTPUT
#{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5411599625818522, 'recall': 0.579, 'accuracy': 0.8342727272727273, 'F1': 0.5605218651848273, 'runtime': 05:16:11

### Investigation Param max depth
investigationList = [None, 2,3,4,5,6,7,8,9,10]
features_list = ['poi', 'shared_receipt_with_poi', 'long_term_incentive', 'bonus', 'frac_from_this_person_to_poi_to_total'] 

for value in investigationList:

    params = {'criterion' : ['entropy'], 
              'splitter' : ['best'], 
              'max_depth': [value],
              'min_samples_split' : [3],
              'min_samples_leaf' : [5],
              'max_features' :[None]}
    
    ### INPUT
    testBestParamsDecisionTree(features_list, my_dataset, params, score='F1')
    ### OUTPUT
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': None, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5611073598919649, 'recall': 0.4205, 'accuracy': 0.8355454545454546, 'F1': 0.48106389129806304, 'runtime': datetime.timedelta(0, 5, 264919)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.54192037470726, 'recall': 0.5785, 'accuracy': 0.8347272727272728, 'F1': 0.5588022216855832, 'runtime': datetime.timedelta(0, 4, 912577)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 3, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.40700218818380746, 'recall': 0.1845, 'accuracy': 0.8025454545454546, 'F1': 0.25077506028246643, 'runtime': datetime.timedelta(0, 5, 74532)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5742372881355933, 'recall': 0.429, 'accuracy': 0.8370909090909091, 'F1': 0.48331415420023016, 'runtime': datetime.timedelta(0, 5, 219545)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 5, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5648648648648649, 'recall': 0.4195, 'accuracy': 0.8352727272727273, 'F1': 0.4808522890872445, 'runtime': datetime.timedelta(0, 5, 427925)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 6, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5733788395904437, 'recall': 0.4135, 'accuracy': 0.8369090909090909, 'F1': 0.48043728423475257, 'runtime': datetime.timedelta(0, 5, 224587)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 7, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5640677966101695, 'recall': 0.42, 'accuracy': 0.8360909090909091, 'F1': 0.4784992784992785, 'runtime': datetime.timedelta(0, 5, 262812)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5707547169811321, 'recall': 0.4195, 'accuracy': 0.8368181818181818, 'F1': 0.4825799021019292, 'runtime': datetime.timedelta(0, 5, 510903)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 9, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5707482993197279, 'recall': 0.4195, 'accuracy': 0.8366363636363636, 'F1': 0.48563218390804597, 'runtime': datetime.timedelta(0, 5, 220959)}
    #{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_features': None}, 'precision': 0.5709436524100475, 'recall': 0.4165, 'accuracy': 0.8357272727272728, 'F1': 0.48242074927953893, 'runtime': datetime.timedelta(0, 5, 216444)}

investigationData = {'max_depth' : ['None', '2','3','4','5','6','7','8','9','10'], 
                     'F1': [0.48106389129806304, 0.5588022216855832, 0.40700218818380746, 0.48331415420023016, 0.4808522890872445, 0.48043728423475257, 0.4784992784992785, 0.4825799021019292, 0.48563218390804597, 0.48242074927953893],
                     'precision' : [0.5611073598919649, 0.54192037470726, 0.40700218818380746, 0.5742372881355933, 0.5648648648648649, 0.5733788395904437, 0.5640677966101695, 0.5707547169811321, 0.5707482993197279, 0.5709436524100475],
                     'recall': [0.4205, 0.5785, 0.1845, 0.429, 0.4195, 0.4135, 0.42, 0.4195, 0.4195, 0.4165],
                     'accuracy' : [0.8355454545454546, 0.8347272727272728, 0.8025454545454546, 0.8370909090909091, 0.8352727272727273, 0.8369090909090909, 0.8360909090909091, 0.8368181818181818,0.8366363636363636, 0.8357272727272728]}
investigationDF = pd.DataFrame(investigationData)

# visualize the investigation data
plt.plot(investigationDF['max_depth'], investigationDF['F1'], color = 'red')
plt.plot(investigationDF['max_depth'], investigationDF['precision'], color = 'blue')
plt.plot(investigationDF['max_depth'], investigationDF['recall'], color = 'yellow')
plt.plot(investigationDF['max_depth'], investigationDF['accuracy'], color = 'green')
plt.xlabel('max depth');
plt.ylabel('values between 0 and 1');
plt.title('Performance Indicator DecisionTree'); 
plt.legend(['F1', 'precision', 'recall', 'accuracy'])
plt.show()

#Params for KNeighbors
params = {'n_neighbors' : [1, 2, 3,4,5,6,7,8,9,10], 
          'weights' : ['uniform', 'distance'], 
          'algorithm': ['auto','ball_tree','kd_tree', 'brute'],
          'p' : [2,3,4,5,6,7,8,9,10]}

features_list = ['poi', 'shared_receipt_with_poi', 'exercised_stock_options', 'bonus', 'frac_bonus_to_total_payments','frac_from_this_person_to_poi_to_total']
### INPUT
#testBestParamsKNeighbors(features_list, my_dataset, params, score='F1')
#{'Type': 'best result', 'Algo': 'DecisionTree', 'params': {'n_neighbors': 6, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.6915167095115681, 'recall': 0.4035, 'accuracy': 0.8805384615384615, 'F1': 0.5096305652036628, 'runtime': datetime.timedelta(0, 5317, 540374)}

### Investigation Param n_neighbors
investigationList = [1, 2,3,4,5,6,7,8,9,10]
features_list = ['poi', 'shared_receipt_with_poi', 'exercised_stock_options', 'bonus', 'frac_bonus_to_total_payments','frac_from_this_person_to_poi_to_total']

for value in investigationList:

    params = {'n_neighbors' : [value], 
              'weights' : ['distance'], 
              'algorithm': ['brute'],
              'p' : [10]}

    ### INPUT
    testBestParamsKNeighbors(features_list, my_dataset, params, score='F1')
    ### OUTPUT
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 1, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.370230607966457, 'recall': 0.4415, 'accuracy': 0.7985384615384615, 'F1': 0.40273660205245154, 'runtime': datetime.timedelta(0, 7, 810668)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 2, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.370230607966457, 'recall': 0.4415, 'accuracy': 0.7985384615384615, 'F1': 0.40273660205245154, 'runtime': datetime.timedelta(0, 7, 415950)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 3, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.4795439302481556, 'recall': 0.3575, 'accuracy': 0.8414615384615385, 'F1': 0.40962474935548554, 'runtime': datetime.timedelta(0, 7, 634478)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 4, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.5527299925205684, 'recall': 0.3695, 'accuracy': 0.857, 'F1': 0.44291279592448307, 'runtime': datetime.timedelta(0, 7, 252448)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.6137339055793991, 'recall': 0.3575, 'accuracy': 0.8665384615384616, 'F1': 0.4518167456556082, 'runtime': datetime.timedelta(0, 7, 178352)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 6, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.6915167095115681, 'recall': 0.4035, 'accuracy': 0.8805384615384615, 'F1': 0.5096305652036628, 'runtime': datetime.timedelta(0, 7, 177799)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 7, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.7155870445344129, 'recall': 0.3535, 'accuracy': 0.8789230769230769, 'F1': 0.47322623828647925, 'runtime': datetime.timedelta(0, 7, 374614)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 8, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.7, 'recall': 0.343, 'accuracy': 0.8763076923076923, 'F1': 0.4604026845637584, 'runtime': datetime.timedelta(0, 7, 213388)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 9, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.7160883280757098, 'recall': 0.3405, 'accuracy': 0.8777692307692307, 'F1': 0.46153846153846156, 'runtime': datetime.timedelta(0, 7, 64042)}
    #{'Type': 'best result', 'Algo': 'KNeighbors', 'params': {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'brute', 'p': 10}, 'precision': 0.7193548387096774, 'recall': 0.3345, 'accuracy': 0.8775384615384615, 'F1': 0.4566552901023891, 'runtime': datetime.timedelta(0, 7, 263449)}

investigationData = {'n_neighbors' : ['1', '2','3','4','5','6','7','8','9','10'], 
                     'F1': [0.40273660205245154, 0.40273660205245154, 0.40962474935548554, 0.44291279592448307, 0.4518167456556082, 0.5096305652036628, 0.47322623828647925, 0.4604026845637584, 0.46153846153846156, 0.4566552901023891],
                     'precision' : [0.370230607966457, 0.370230607966457, 0.4795439302481556, 0.5527299925205684, 0.6137339055793991, 0.6915167095115681, 0.7155870445344129, 0.7, 0.7160883280757098, 0.7193548387096774],
                     'recall': [0.4415, 0.4415, 0.3575, 0.3695, 0.3575, 0.4035, 0.3535, 0.343, 0.3405, 0.3345],
                     'accuracy' : [0.7985384615384615, 0.7985384615384615, 0.8414615384615385, 0.857, 0.8665384615384616, 0.8805384615384615, 0.8789230769230769, 0.8763076923076923, 0.8777692307692307, 0.8775384615384615]}
investigationDF = pd.DataFrame(investigationData)

# visualize the investigation data
plt.plot(investigationDF['n_neighbors'], investigationDF['F1'], color = 'red')
plt.plot(investigationDF['n_neighbors'], investigationDF['precision'], color = 'blue')
plt.plot(investigationDF['n_neighbors'], investigationDF['recall'], color = 'yellow')
plt.plot(investigationDF['n_neighbors'], investigationDF['accuracy'], color = 'green')
plt.xlabel('n_neighbors');
plt.ylabel('values between 0 and 1');
plt.title('Performance Indicator KNeighbors'); 
plt.legend(['F1', 'precision', 'recall', 'accuracy'])
plt.show()
# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split ''' old Version Py 2
from sklearn.model_selection import train_test_split

features_list = ['poi', 'shared_receipt_with_poi', 'long_term_incentive', 'bonus', 'frac_from_this_person_to_poi_to_total'] 
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = 2, min_samples_split = 3, min_samples_leaf = 5, max_features = None)

clf.fit(features_train, labels_train)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)