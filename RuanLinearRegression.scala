package RuanPackage

import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql._
import org.apache.spark.sql.types._

object RuanLinearRegression {
  
  case class RegressionSchema(label: Double, PageSpeed: Double)

  /** main function*/
  def main(args: Array[String]) {
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    //Creating Spark Session
    val spark = SparkSession
      .builder
      .appName("RuanLinearRegression")
      .master("local[*]")
      .getOrCreate()

    // Load up our amount spent and page speed data in the format required by MLLib
    // (which is label, vector of features)
    val regressionSchema = new StructType()
      .add("label", DoubleType, nullable = true)
      .add("PageSpeed", DoubleType, nullable = true)

    //Importing dataset
    import spark.implicits._
    val dsRaw = spark.read
      .option("sep", ",")
      .schema(regressionSchema)
      .csv("data/regression.txt")
      .as[RegressionSchema]

    //Creating Dataframe that is required for this MLLib
    val assembler = new VectorAssembler().
      setInputCols(Array("PageSpeed")).
      setOutputCol("features")
    val df = assembler.transform(dsRaw)
      .select("label","features")

    // Splitting the data into training data and testing data
    val trainTest = df.randomSplit(Array(0.5, 0.5))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)

    // Using the linear regression model
    val lir = new LinearRegression()
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setMaxIter(100)
      .setTol(1E-6)

    // Train the model using our training data
    val model = lir.fit(trainingDF)

    // Now the model tries to predict the values
    val fullPredictions = model.transform(testDF).cache()

    // This basically adds a "prediction" column to our testDF dataframe.

    // Extract the predictions and the correct labels.
    val predictionAndLabel = fullPredictions.select("prediction", "label").collect()

    // Print out the predicted and actual values for each point
    for (prediction <- predictionAndLabel) {
      println(prediction)
    }

    // Stop the session
    spark.stop()

  }
}
