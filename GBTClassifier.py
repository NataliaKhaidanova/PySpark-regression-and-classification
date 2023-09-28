from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
    gbt = GBTClassifier()
    gbt.fit(flights_train)
    
    # Get AUC on test data
    evaluator = BinaryClassificationEvaluator()
    predictions = gbt.transform(flights_test)
    print('AUC of the default GBTClassifier = {}\n' .format(evaluator.evaluate(predictions)))

    # Find the number of trees and the relative importance of features
    # print(gbt.trees)
    print(gbt.featureImportances, '\n')

    # Perfom hyperparameter tuning 
    params = ParamGridBuilder()

    params = params.addGrid(gbt.maxDepth, [3, 5, 7]) \
                   .addGrid(gbt.maxBins, [32, 64, 128]) \
                   .addGrid(gbt.maxIter, [10, 20, 50]) \
                   .addGrid(gbt.stepSize, [0.01, 0.1, 0.2]) \
                   .addGrid(gbt.impurity, ['entropy', 'gini', 'variance']) \
                   .build() 

    # Other hyperparameters to consider: 
    # minInstancesPerNode, [1, 2, 5]
    # minInfoGain, [0.0, 0.01, 0.1]) 
    # subsamplingRate, [0.7, 0.8, 1.0]
    # featureSubsetStrategy, ['all', 'onethird', 'sqrt']
    # validationTol, [0.001, 0.01, 0.1]

    # Perform cross-validation 
    gbt = GBTClassifier() 
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

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
    # Best Param (maxIter): 650 (default: 100, current: 650)
    # Best Param (regParam): 0.0 (default: 0.0, current: 0.0)
    # Best Param (elasticNetParam): 0.0 (default: 0.0, current: 0.0)
    # Best Param (fitIntercept): False (default: True, current: False)

    predictions = best_model.transform(flights_test)
    print('AUC of the best GBTClassifier = {}\n' .format(evaluator.evaluate(predictions)))
    # print(gbt.trees)
    print(gbt.featureImportances)

    zero_coeff = len([beta for beta in best_model.coefficients if beta == 0])
    print('Number of coefficients equal to 0: {}\n' .format(int(zero_coeff)))

    # RESULT:
    # RMSE of the best model = 7.671071994215025e-14
    # Number of coefficients equal to 0: 0
