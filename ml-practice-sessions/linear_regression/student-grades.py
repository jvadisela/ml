from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('student_grades.csv', header = True, inferSchema = True)

#Taking a look at data type of each column to see what data types inferSchema=TRUE paramter has set for each column
df.printSchema()
df.show()


#Create a Feature array by omitting the last column
feature_cols = df.columns[:-1]

from pyspark.ml.feature import VectorAssembler
vect_assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")

#Utilize Assembler created above in order to add the feature column
data_w_features = vect_assembler.transform(df)

#Display the data having additional column named features. Had it been multiple linear regression problem, you could see all the
# independent variable values combined in one list
data_w_features.show()

#Select only Features and Label from previous dataset as we need these two entities for building machine learning model
finalized_data = data_w_features.select("features","Grades")
finalized_data.show()

#Split the data into training and test model with 70% obs. going in training and 30% in testing
train_dataset, test_dataset = finalized_data.randomSplit([0.7, 0.3])

#Peek into training data
train_dataset.describe().show()

#Peek into test_dataset
test_dataset.describe().show()

#Import Linear Regression class called LinearRegression
from pyspark.ml.regression import LinearRegression

#Create the Linear Regression object named having feature column as features and Label column as Time_to_Study
LinReg = LinearRegression(featuresCol="features", labelCol="Grades")

#Train the model on the training using fit() method.
model = LinReg.fit(train_dataset)

#Predict the Grades using the evulate method
pred = model.evaluate(test_dataset)

#Show the predicted Grade values along side actual Grade values
pred.predictions.show()

#Find out coefficient value
coefficient = model.coefficients
print ("The coefficient of the model is : %a", coefficient)

#Find out intercept Value
intercept = model.intercept
print ("The Intercept of the model is : %f" %intercept)

#Evaluate the model using metric like Mean Absolute Error(MAE), Root Mean Square Error(RMSE) and R-Square
from pyspark.ml.evaluation import RegressionEvaluator
evaluation = RegressionEvaluator(labelCol="Grades", predictionCol="prediction")

# Root Mean Square Error
rmse = evaluation.evaluate(pred.predictions, {evaluation.metricName: "rmse"})
print("RMSE: %.3f" % rmse)

# Mean Square Error
mse = evaluation.evaluate(pred.predictions, {evaluation.metricName: "mse"})
print("MSE: %.3f" % mse)

# Mean Absolute Error
mae = evaluation.evaluate(pred.predictions, {evaluation.metricName: "mae"})
print("MAE: %.3f" % mae)

# r2 - coefficient of determination
r2 = evaluation.evaluate(pred.predictions, {evaluation.metricName: "r2"})
print("r2: %.3f" %r2)

#Create Unlabeled dataset  to contain only feature column
unlabeled_dataset = test_dataset.select('features')

#Display the content of unlabeled_dataset
unlabeled_dataset.show()

#Predict the model output for fresh & unseen test data using transform() method
new_predictions = model.transform(unlabeled_dataset)

#Display the new prediction values
new_predictions.show()