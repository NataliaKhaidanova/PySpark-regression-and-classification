from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler

# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('flight_delay') \
                    .getOrCreate()
                    
flights = spark.read.csv('flights-larger.csv', sep=',', header=True, inferSchema=True, nullValue='NA')

print('The data contains %d records.' % flights.count())

flights = flights.drop('flight') # Remove the "flight" column

# All missing values come from the "delay" column (as stated from data analysis) 
print('The data contains %d missing values.' % flights.filter('delay IS NULL').count()) 
# Remove records with missing "delay" values 
flights = flights.filter('delay IS NOT NULL') 
# Or remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna() 
print('The data contains %d records after removing the missing values.' % flights.count())

# Convert "mile" to "km" and drop "mile" column (1 mile is equivalent to 1.60934 km)
flights = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                 .drop('mile')

# Create "label" column indicating whether flight was delayed (1) or not (0)
flights = flights.withColumn('label', (when (flights.delay >= 15, 1)
.otherwise (0)).cast('integer'))

# Create an indexer, create a new column with numeric index values for string data
flights = StringIndexer(inputCol='carrier', outputCol='carrier_idx').fit(flights).transform(flights)
flights = StringIndexer(inputCol='org', outputCol='org_idx').fit(flights).transform(flights)
flights.show(5)

# Create an instance of the one hot encoder
onehot = OneHotEncoder(inputCols=['carrier_idx','org_idx'], outputCols=['carrier_dummy','org_dummy'])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
onehot = onehot.transform(flights)

# Check the results
onehot.select('org', 'org_idx', 'org_dummy').distinct().orderBy('org_idx').show()
onehot.select('carrier', 'carrier_idx', 'carrier_dummy').distinct().orderBy('carrier_idx').show()

