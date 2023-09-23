from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from utils import *

flights_train, flights_test = train_test_split('regression')

for model in ['default', 'Lasso']: # default = Ridge-like
    if model == 'default':
        # Create a regression object and train on training data
        regression = LinearRegression(labelCol='duration').fit(flights_train)
    else:
        # Fit Lasso model (λ = 1, α = 1) to training data
        regression = LinearRegression(labelCol='duration', regParam=1, elasticNetParam=1).fit(flights_train)

    # Create predictions for the testing data and take a look at the predictions
    predictions = regression.transform(flights_test)
    #predictions.select('duration', 'prediction').show(5, False) 

    # Calculate the RMSE on testing data
    rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
    print('RMSE of the {} model = {}' .format(model, rmse))

    # Number of zero coefficients
    zero_coeff = len([beta for beta in regression.coefficients if beta == 0])
    print("Number of coefficients equal to 0: {}\n" .format(int(zero_coeff)))
