# Classification - AUC
# Logistic Regression with cross-validation and hyperparameter tuning 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from utils import *

 
flights_train, flights_test = train_test_split('classification')


def logistic_regression(flights_train, flights_test):
    """
    Train and test (using AUC) LogisticRegression model,
    Perform hyperparameter tuning and cross-validation, 
    Evaluate the best model,
    Compare the default model to the best model. 
    
    :param list flights_train: list of training data with features 
    :param list flights_test: list of test data with features 
    :return: none
    """
    logistic = LogisticRegression().fit(flights_train)

    # Get AUC on test data
    evaluator = BinaryClassificationEvaluator()
    predictions = logistic.transform(flights_test)
    print('AUC of the default LogisticRegression = {}\n' .format(evaluator.evaluate(predictions)))

    # RESULT:    
    # AUC of the default LogisticRegression = 0.6492130116838126
                                          
    # Perfom hyperparameter tuning 
    params = ParamGridBuilder()

    params = params.addGrid(logistic.maxIter, [10, 50, 100]) \
                   .addGrid(logistic.regParam, [0.01, 0.1, 1.0]) \
                   .addGrid(logistic.elasticNetParam, [0.0, 0.5, 1.0]) \
                   .build()

    # Perform cross-validation 
    logistic = LogisticRegression() 
    cv = CrossValidator(estimator=logistic, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

    cv = cv.fit(flights_train)

    # Extract the best model 
    best_model = cv.bestModel

    # Look at the hyperparameters in the best model
    print('Best Param (maxIter):', best_model.getMaxIter(), best_model.explainParam('maxIter'))
    print('Best Param (regParam):', best_model.getRegParam(), best_model.explainParam('regParam'))
    print('Best Param (elasticNetParam):', best_model.getElasticNetParam(), best_model.explainParam('elasticNetParam'))

    # RESULT: 
    # Best Param (maxIter): 100 (default: 100)
    # Best Param (regParam): 0.0 (default: 0.0)
    # Best Param (elasticNetParam): 0.0 (default: 0.0)

    predictions = best_model.transform(flights_test)
    print('\nAUC of the best LogisticRegression = {}\n' .format(evaluator.evaluate(predictions)))

    # RESULT:
    # AUC of the best LogisticRegression = 0.6492098424114412


if __name__ == '__main__':
    logistic_regression(flights_train, flights_test)
