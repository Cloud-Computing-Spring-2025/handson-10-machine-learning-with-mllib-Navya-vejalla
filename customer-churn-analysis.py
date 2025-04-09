from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import os
# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

def save_pretty_output(df, file_path, num_rows=5):
    with open(file_path, "w") as f:
        f.write("+--------------------+-----------+\n")
        f.write("|features            |ChurnIndex |\n")
        f.write("+--------------------+-----------+\n")
        for row in df.select("features", "label").take(num_rows):
            f.write(f"|{str(row.features):<20}|{row.label:<11}|\n")
        f.write("+--------------------+-----------+\n")

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # Fill missing values
    # Encode categorical variables    
    # One-hot encode indexed features
    # Assemble features into a single vector 
    df = df.fillna({'TotalCharges': 0})

    categorical_cols = ["gender", "PhoneService", "InternetService", "Churn"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "Index", outputCol=col + "Vec") for col in categorical_cols[:-1]]

    for indexer in indexers:
        df = indexer.fit(df).transform(df)
    for encoder in encoders:
        df = encoder.fit(df).transform(df)

    feature_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + [col + "Vec" for col in categorical_cols[:-1]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    df = df.withColumnRenamed("ChurnIndex", "label")

    os.makedirs("output/task1", exist_ok=True)
    save_pretty_output(df, "output/task1/processed_sample.txt")

    return df.select("features", "label")


# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    # Split data into training and testing sets
    # Train logistic regression model
    # Predict and evaluate
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)

    os.makedirs("output/task2", exist_ok=True)
    with open("output/task2/logistic_regression_results.txt", "w") as f:
        f.write(f"Logistic Regression Model Accuracy (AUC): {auc:.2f}\n")

    save_pretty_output(predictions, "output/task2/predictions_sample.txt")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
   
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)

    os.makedirs("output/task3", exist_ok=True)
    with open("output/task3/selected_features_sample.txt", "w") as f:
        f.write("+--------------------+-----------+\n")
        f.write("|selectedFeatures    |ChurnIndex |\n")
        f.write("+--------------------+-----------+\n")
        for row in result.select("selectedFeatures", "label").take(5):
            f.write(f"|{str(row.selectedFeatures):<20}|{row.label:<11}|\n")
        f.write("+--------------------+-----------+\n")


# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    # Split data
    # Define models
    # Define hyperparameter grids
    # Perform cross-validation for each model
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="label")

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
        "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label")
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).addGrid(models["LogisticRegression"].maxIter, [10, 20]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [5, 10]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].maxDepth, [10, 15]).addGrid(models["RandomForest"].numTrees, [20, 50]).build(),
        "GBT": ParamGridBuilder().addGrid(models["GBT"].maxDepth, [5, 10]).addGrid(models["GBT"].maxIter, [10, 20]).build()
    }

    os.makedirs("output/task4", exist_ok=True)
    results_file = open("output/task4/model_comparison_results.txt", "w")

    for name, model in models.items():
        results_file.write(f"Tuning {name}...\n")
        grid = param_grids[name]
        cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        results_file.write(f"{name} Best Model Accuracy (AUC): {auc:.2f}\n")
        tuned_params = [param.name for param in grid[0].keys()]  # get tuned param names
        best_params = best_model.extractParamMap()
        param_str = ", ".join([
            f"{param.name}={value}" 
            for param, value in best_params.items() 
            if param.name in tuned_params and param.parent == best_model.uid
        ])
        results_file.write(f"Best Params for {name}: {param_str}\n\n")

        # Save sample predictions
        with open(f"output/task4/{name}_predictions_sample.txt", "w") as pred_file:
            pred_file.write("+--------------------+-----------+\n")
            pred_file.write("|features            |prediction |\n")
            pred_file.write("+--------------------+-----------+\n")
            for row in predictions.select("features", "prediction").take(5):
                pred_file.write(f"|{str(row.features):<20}|{row.prediction:<11}|\n")
            pred_file.write("+--------------------+-----------+\n")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()