from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from utils import *

flights_train, flights_test = train_test_split('classification')

def GBT_classifier(flights_train, flights_test):
    """
    Train and test (using AUC) GBTClassifier model,
    Perform hyperparameter tuning and cross-validation, 
    Evaluate and save the best model,
    Compare the default model to the best model. 
    
    :param list flights_train: list of training data with features 
    :param list flights_test: list of test data with features 
    :return: none
    """
    gbt = GBTClassifier().fit(flights_train)

    # Get AUC on test data
    evaluator = BinaryClassificationEvaluator()
    predictions = gbt.transform(flights_test)
    print('AUC of the default GBTClassifier = {}\n' .format(evaluator.evaluate(predictions)))

    # Find the number of trees and the relative importance of features
    #print(gbt.trees)
    print(gbt.featureImportances, '\n')

    # RESULT:    
    # AUC of the default GBTClassifier = 0.7296886919893384

    # Perfom hyperparameter tuning 
    params = ParamGridBuilder()

    params = params.addGrid(gbt.maxDepth, [5, 7]) \
                   .addGrid(gbt.maxBins, [32, 64, 128]) \
                   .addGrid(gbt.maxIter, [20, 50]) \
                   .addGrid(gbt.stepSize, [0.01, 0.1]) \
                   .addGrid(gbt.impurity, ['entropy', 'gini', 'variance']) \
                   .build() 

    # Other hyperparameters to consider: 
    # minInstancesPerNode, [1, 2, 3]
    # minInfoGain, [0.0, 0.01, 0.1]) 
    # subsamplingRate, [0.8, 0.9, 1.0]
    # featureSubsetStrategy, ['all', 'onethird', 'sqrt']
    # validationTol, [0.001, 0.01, 0.1]

    # Perform cross-validation 
    gbt = GBTClassifier() 
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=params, evaluator=evaluator, numFolds=3)

    cv = cv.fit(flights_train)

    # Extract the best model 
    best_model = cv.bestModel

    # Look at the hyperparameters in the best model
    print('Best Param (maxDepth):', best_model.getMaxDepth(), best_model.explainParam('maxDepth'))
    print('Best Param (maxBins):', best_model.getMaxBins(), best_model.explainParam('maxBins'))
    print('Best Param (maxIter):', best_model.getMaxIter(), best_model.explainParam('maxIter'))
    print('Best Param (stepSize):', best_model.getStepSize(), best_model.explainParam('stepSize'))
    print('Best Param (impurity):', best_model.getImpurity(), best_model.explainParam('impurity'))

    # RESULT: 
    # Best Param (maxDepth): 5 (default: 5)
    # Best Param (maxBins): 32 (default: 32)
    # Best Param (maxIter): 20 (default: 20)
    # Best Param (stepSize): 0.1 (default: 0.1)
    # Best Param (impurity): variance impurity (default: variance)

    predictions = best_model.transform(flights_test)
    print('\nAUC of the best GBTClassifier = {}\n' .format(evaluator.evaluate(predictions)))
    #print(gbt.trees)
    print(best_model.featureImportances)

    # RESULT:
    # AUC of the best GBTClassifier = 0.7267226346255101
