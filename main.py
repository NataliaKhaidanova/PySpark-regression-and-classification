from pyspark.sql import SparkSession

# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('flight_delay') \
                    .getOrCreate()
                    
flights = spark.read.csv('flights-larger.csv', sep=',', header=True, inferSchema=True, nullValue='NA')

print("The data contains %d records." % flights.count())

flights = flights.drop('flight') # Remove the "flight" column

# All missing values come from the "delay" column (as stated from data analysis) 
print("The data contains %d missing values." % flights.filter('delay IS NULL').count()) 
# Remove records with missing "delay" values 
flights = flights.filter('delay IS NOT NULL') 
# Or remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna() 
print("The data contains %d records after removing the missing values." % flights.count())

