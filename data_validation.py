import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("FraudDetection") \
    .getOrCreate()

def load_data(file_path):
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    
    data = data.na.fill(0)
    
    if 'credit_card_number' in data.columns:
        data = data.withColumn('credit_card_number', col('credit_card_number').cast('string'))
    
    return data
