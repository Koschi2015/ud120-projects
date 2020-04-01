from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import StratifiedShuffleSplit
import datetime

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return clf, accuracy, precision, recall, f1, f2, total_predictions, true_positives, false_positives, false_negatives, true_negatives, feature_list
        #print(clf)
        #print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        #print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        #print("")
    except:
        return clf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
def getListValues(allListValues, number):
    keyList = []
    for i in range(len(allListValues)):
        # check for power of 2 and add Feature
        if (number & 2**i) > 0:
            keyList.append(allListValues[i])
        
    return keyList


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def getBestFeaturesResult(algoNames, features_list_all, my_dataset, score = 'F1'):
    resultList = {}
    for eachAlgo in algoNames:
        startTime = datetime.datetime.now()
        bestResult = {'Type' : 'best result', 'Algo':eachAlgo, 'featureList' : [] , 'precision': 0, 'recall': 0,'accuracy': 0, 'F1': 0,'runtime' : 0}
        counter = 0
        for i in range(2**len(features_list_all)-1):
            features_list = ['poi']
            for each in getListValues(features_list_all,i + 1):
                features_list.append(each)
            
            data = featureFormat(my_dataset, features_list, sort_keys = True)
            
            labels, features = targetFeatureSplit(data)
            if eachAlgo == 'DecisionTree':
                clf = DecisionTreeClassifier()
            elif eachAlgo == 'KNeighbors':
                clf = KNeighborsClassifier()
            elif eachAlgo == 'GaussianNB':
                clf = GaussianNB()
                
            clf.fit(features, labels)
            precision = test_classifier(clf, my_dataset, features_list)[2]
            recall = test_classifier(clf, my_dataset, features_list)[3]
            accuracy = test_classifier(clf, my_dataset, features_list)[1]
            F1 = test_classifier(clf, my_dataset, features_list)[4]
            
            if score == 'F1':
                measuredValue = F1
            elif score == 'accuracy':
                measuredValue = accuracy
            elif score == 'precision':
                measuredValue = precision
            elif score == 'recall':
                measuredValue = recall
                
            if measuredValue >= bestResult[score]:
                bestResult['featureList'] = features_list[1:]
                bestResult['precision'] = precision
                bestResult['recall'] = recall
                bestResult['accuracy'] = accuracy
                bestResult['F1'] = F1
            
            counter = counter + 1
            
        bestResult['runtime'] =  datetime.datetime.now() - startTime
        resultList[eachAlgo] = bestResult           
 
    print('| algorithms | recall | precision | accuracy | F1 | features | runtime | ')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    for each in resultList:
        print('|', each, '|', resultList[each]['recall'], '|', resultList[each]['precision'], '|', resultList[each]['accuracy'], '|', resultList[each]['F1'], '|', resultList[each]['featureList'], '|', resultList[each]['runtime'], '|')
    
def testBestParamsDecisionTree(features_list, my_dataset, params, score):
    startTime = datetime.datetime.now()
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    bestResult = {'Type' : 'best result', 'Algo': 'DecisionTree', 'params' : {'criterion' : None,
                                                                              'splitter': None,
                                                                              'max_depth': None,
                                                                              'min_samples_split' : None,
                                                                              'min_samples_leaf' : None,
                                                                              'max_features' : None} , 'precision': 0, 'recall': 0,'accuracy': 0, 'F1': 0,'runtime' : 0}

    for criterionParam in params['criterion']:
        for splitterParam in params['splitter']:
            for max_depthmParam in params['max_depth']:
                for min_samples_splitParam in params['min_samples_split']:
                    for min_samples_leafParam in params['min_samples_leaf']:
                        for max_featuresParam in params['max_features']:
                            clf = DecisionTreeClassifier(criterion = criterionParam, splitter = splitterParam, max_depth = max_depthmParam, min_samples_split = min_samples_splitParam, min_samples_leaf = min_samples_leafParam, max_features = max_featuresParam)
                            clf.fit(features, labels)
                            precision = test_classifier(clf, my_dataset, features_list)[2]
                            recall = test_classifier(clf, my_dataset, features_list)[3]
                            F1 = test_classifier(clf, my_dataset, features_list)[4]
                            accuracy = test_classifier(clf, my_dataset, features_list)[1]
                            
                            if score == 'F1':
                                measuredValue = F1
                            elif score == 'accuracy':
                                measuredValue = accuracy
                            elif score == 'precision':
                                measuredValue = precision
                            elif score == 'recall':
                                measuredValue = recall
                                
                            if measuredValue >= bestResult[score]:
                                bestResult['precision'] = precision
                                bestResult['recall'] = recall
                                bestResult['accuracy'] = accuracy
                                bestResult['F1'] = F1
                                bestResult['params']['criterion'] = criterionParam
                                bestResult['params']['splitter'] = splitterParam
                                bestResult['params']['max_depth'] = max_depthmParam
                                bestResult['params']['min_samples_split'] = min_samples_splitParam
                                bestResult['params']['min_samples_leaf'] = min_samples_leafParam
                                bestResult['params']['max_features'] = max_featuresParam
    
    bestResult['runtime'] = datetime.datetime.now() - startTime
    
    print(bestResult)
    
def testBestParamsKNeighbors(features_list, my_dataset, params, score):
    startTime = datetime.datetime.now()
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    bestResult = {'Type' : 'best result', 'Algo': 'KNeighbors', 'params' : {'n_neighbors' : None,
                                                                              'weights': None,
                                                                              'algorithm': None,
                                                                              'p' : None,} , 'precision': 0, 'recall': 0,'accuracy': 0, 'F1': 0,'runtime' : 0}

    for n_neighborsParam in params['n_neighbors']:
        for weightsParam in params['weights']:
            for algorithmParam in params['algorithm']:
                for pParam in params['p']:
                    clf = KNeighborsClassifier(n_neighbors = n_neighborsParam, weights = weightsParam, algorithm = algorithmParam, p = pParam)
                    clf.fit(features, labels)
                    precision = test_classifier(clf, my_dataset, features_list)[2]
                    recall = test_classifier(clf, my_dataset, features_list)[3]
                    F1 = test_classifier(clf, my_dataset, features_list)[4]
                    accuracy = test_classifier(clf, my_dataset, features_list)[1]
                    
                    if score == 'F1':
                        measuredValue = F1
                    elif score == 'accuracy':
                        measuredValue = accuracy
                    elif score == 'precision':
                        measuredValue = precision
                    elif score == 'recall':
                        measuredValue = recall
                                
                    if measuredValue >= bestResult[score]:
                        bestResult['precision'] = precision
                        bestResult['recall'] = recall
                        bestResult['accuracy'] = accuracy
                        bestResult['F1'] = F1
                        bestResult['params']['n_neighbors'] = n_neighborsParam
                        bestResult['params']['weights'] = weightsParam
                        bestResult['params']['algorithm'] = algorithmParam
                        bestResult['params']['p'] = pParam

    bestResult['runtime'] = datetime.datetime.now() - startTime
                        
    print(bestResult)

