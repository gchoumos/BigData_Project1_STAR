
// Read data 
val jobs = spark.sqlContext.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .option("quote","\"")
    .option("escape","\"")
    .load("fake_job_postings.csv")


// Number of lines
jobs.count()


// Number of fake job postings
jobs.filter("fraudulent = 1").count()


// Number of real job postings
jobs.filter("fraudulent = 0").count()


// Show top 10 most required education in fake job postings
jobs.filter("fraudulent == 1")
    .groupBy("required_education")
    .count()
    .orderBy($"count".desc)
    .show(10,false)


// Show top 10 most required education in real job postings
jobs.filter("fraudulent == 0")
    .groupBy("required_education")
    .count()
    .orderBy($"count".desc)
    .show(10,false)

