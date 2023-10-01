from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from utils import *

flights_train, flights_test = train_test_split('regression')

def linear_regression(flights_train, flights_test):
    """
    Train and test (using RMSE) LinearRegression model,
    Compare the default model to its Lasso and Ridge versions,
    Perform hyperparameter tuning and cross-validation, 
    Evaluate and save the best model. 
    
    :param list flights_train: list of training data with features 
    :param list flights_test: list of test data with features 
    :return: none
    """
    # COMPARE DEFAULT, LASSO, AND RIDGE LINEAR REGRESSION 
    for model in ['default', 'Lasso', 'Ridge']: # default = Ridge-like
        if model == 'default':
            # Create a regression object and train it on training data
            regression = LinearRegression(labelCol='duration').fit(flights_train)
        if model == 'Lasso':
            # Fit Lasso model (λ = 1, α = 1) to training data
            regression = LinearRegression(labelCol='duration', regParam=1, elasticNetParam=1).fit(flights_train)
        else:
            # Fit Ridge model (λ = 0, α = 0) to training data
            regression = LinearRegression(labelCol='duration', regParam=0, elasticNetParam=0).fit(flights_train)

        # Create predictions for the testing data 
        predictions = regression.transform(flights_test)
        #predictions.select('duration', 'prediction').show(5, False) # Take a look at the predictions

        # Calculate the RMSE on test data
        rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
        print('RMSE of the {} LinearRegression model = {}' .format(model, rmse))

        zero_coeff = len([beta for beta in regression.coefficients if beta == 0])
        print('Number of coefficients equal to 0: {}\n' .format(int(zero_coeff)))

    # RESULT:    
    # RMSE of the default model = 9.178962994062984e-13
    # Number of coefficients equal to 0: 0

    # RMSE of the Lasso model = 0.9966009994133522
    # Number of coefficients equal to 0: 20

    # RMSE of the Ridge model = 9.178962994062984e-13
    # Number of coefficients equal to 0: 0

    # PERFORM HYPERPARAMETER TUNING AND CROSS-VALIDATION 
    regression = LinearRegression(labelCol='duration')
    evaluator = RegressionEvaluator(labelCol='duration')

    # Perfom hyperparameter tuning 
    params = ParamGridBuilder()

    params = params.addGrid(regression.maxIter, [550, 600, 650]) \
                   .addGrid(regression.regParam, [0.0, 0.001, 0.01]) \
                   .addGrid(regression.elasticNetParam, [0.0, 0.3, 0.5, 0.8, 1.0]) \
                   .addGrid(regression.fitIntercept, [True, False]) \
                   .build()

    # Perform cross-validation 
    cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

    cv = cv.fit(flights_train)

    # Extract the best model 
    best_model = cv.bestModel
    best_model.save('models/LinearRegression')

    # Look at the hyperparameters in the best model
    print('Best Param (maxIter):', best_model.getMaxIter(), best_model.explainParam('maxIter'))
    print('Best Param (regParam):', best_model.getRegParam(), best_model.explainParam('regParam'))
    print('Best Param (elasticNetParam):', best_model.getElasticNetParam(), best_model.explainParam('elasticNetParam'))
    print('Best Param (fitIntercept):', best_model.getFitIntercept(), best_model.explainParam('fitIntercept'))

    # RESULT: 
    # Best Param (maxIter): 650 (default: 100, current: 650)
    # Best Param (regParam): 0.0 (default: 0.0, current: 0.0)
    # Best Param (elasticNetParam): 0.0 (default: 0.0, current: 0.0)
    # Best Param (fitIntercept): False (default: True, current: False)

    predictions = best_model.transform(flights_test)
    print('\nRMSE of the best LinearRegression model = {}' .format(evaluator.evaluate(predictions)))

    zero_coeff = len([beta for beta in best_model.coefficients if beta == 0])
    print('Number of coefficients equal to 0: {}\n' .format(int(zero_coeff)))

    # RESULT:
    # RMSE of the best model = 7.671071994215025e-14
    # Number of coefficients equal to 0: 0
