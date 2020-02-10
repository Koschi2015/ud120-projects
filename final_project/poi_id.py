#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

#from feature_format import featureFormat, targetFeatureSplit ''' old Version Py 2 '''
#from tester import dump_classifier_and_data ''' old Version Py 2 '''
from tools.feature_format import featureFormat, targetFeatureSplit
from final_project.tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "r") as data_file: ''' old Version Py2'''
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
        
with open("final_project_dataset_new.pkl", 'rb') as data_file2:        
    data_dict = pickle.load(data_file2)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split ''' old Version Py 2
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)