
# Read the data. You can find documentation on spark.read.csv here:
# https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=read%20csv
jobs = spark.read.csv(
    "fake_job_postings.csv",
    header=True,
    sep=",",
    escape="\"",
    inferSchema=True
)

# Remove rows with null values
jobs = jobs.dropna()

# The previous is expected to have reduced the dataframe size significantly. Indeed..
jobs.count()

# Worth creating new columns for minimum and maximum salary
# Will do that with user defined functions for min and max
import numpy as np
import matplotlib.pyplot as plt
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType

def getMinSalary(value):
    return int(value.split('-')[0])

def getMaxSalary(value):
    return int(value.split('-')[1])

# Convert the above to udf functions
udf_getMinSalary = f.udf(getMinSalary, IntegerType())
udf_getMaxSalary = f.udf(getMaxSalary, IntegerType())

# Add the new columns for min and max and then drop the range as it not needed anymore
jobs = jobs.withColumn("min_salary",udf_getMinSalary("salary_range").cast(IntegerType()))
jobs = jobs.withColumn("max_salary",udf_getMaxSalary("salary_range").cast(IntegerType()))

# Then drop the salary_range column that is now redundant
jobs = jobs.drop("salary_range")

# Compute the average max_salary for the fake (fraudulent == 1) job postings
# Another way to do this is:
#  jobs.filter("fraudulent == 1").groupBy().avg("max_salary").show()
jobs.filter("fraudulent == 1").agg({'max_salary': 'mean'}).show()

# Compute the standard deviation of max_salary for the fake job postings
jobs.filter("fraudulent == 1").agg({'max_salary': 'stddev'}).show()

# Compute the median. We have 702 real job postings with no null values. Thus, to compute the
# median we will get the average of rows 351 and 352. Note that we actually use indexes 350 and
# and 351 because indexing in Python starts at 0 (as is the case with most programming languages).
salary1 = jobs.filter("fraudulent == 0").select("min_salary").sort("min_salary").collect()[350][0]
salary2 = jobs.filter("fraudulent == 0").select("min_salary").sort("min_salary").collect()[351][0]
print("Median of min salary for real job postings is {0}".format((salary1+salary2)/2))


# A common first step to detect outliers is to just look for values that stand out.
# Let's have a look at the top 5 minimum salaries and their max_salary counterpant.
jobs.select('min_salary','max_salary').orderBy('min_salary',ascending=False).show(5)

# Check how many cases exist with zeroes in either or both the min_salary and max_salary.
# Then check what are the distinct cases out of curiosity. Then drop them.
jobs.select('min_salary','max_salary').filter('min_salary = 0 or max_salary = 0').count()
jobs.select('min_salary','max_salary').filter('min_salary = 0 or max_salary = 0').distinct().show()
jobs = jobs.filter('min_salary > 0 and max_salary > 0')
jobs.count()

# Quick sanity check that all min_salary are less than or equal to their max_salary counterpart
jobs.filter('min_salary <= max_salary').count()

# Let's check a histogram of the min_salary and max_salary
pd_jobs = jobs.toPandas()
plt.hist(np.log10(pd_jobs.min_salary.values),bins=100,alpha=0.5,label='min salary')
plt.hist(np.log10(pd_jobs.max_salary.values),bins=100,alpha=0.5,label='max salary')
plt.xlabel('Logarithm of min and max salary',fontsize=15)
plt.ylabel('Occurences',fontsize=15)
plt.legend(loc='upper left')
plt.show()

# Based on the above histogram, we'll consider outliers any rows for which
# log(min_salary) <= 3.5, thus, min_salary <= 3162
# or rows for which log(max_salary) >= 6.5, thus, max_salary >= 3162277
# Let's see a breakdown of the affected rows and how many are fraudulent
jobs.filter('min_salary < 3162 or max_salary > 3162277').groupBy('fraudulent').count().show()

# None of the affected rows was fraudulent

jobs = jobs.filter('min_salary > 3162 and max_salary < 3162277')
jobs.count()

# Now recompute the average and standard deviation of maximum salary - Fake Job Postings
jobs.filter("fraudulent == 1").agg({'max_salary': 'mean'}).show()


# Re-Compute the median. After removing the outliers we now have 679 real job postings. So
# the median will be the value of row 340, thus, index 339 (because Python indexing starts at 0)
new_median = jobs.filter("fraudulent == 0").select("min_salary").sort("min_salary").collect()[339][0]
print("New median of min salary for real job postings is {0}".format(new_median))

# It's the last question of the section and we only need the description and fraudulent.
# Let's keep only these 2 columns and also convert them to a Pandas dataframe.
jobs = jobs.select('description','fraudulent').toPandas()

# Clean up a bit by replacing the following with a space. They first 3 are just masked values
# and they are not useful for the n-grams statistics.
#   '#EMAIL_*#'
#   '#PHONE_*#'
#   '#URL_*#'
#   '\xa0'
#   '&amp;' and similar HTML character references
jobs[['description']] = jobs[['description']].replace([r'#(URL|EMAIL|PHONE)_\S*#'],' ', regex=True)
jobs[['description']] = jobs[['description']].replace([r'\xa0',r'&\w+;'],' ', regex=True)

# Now also replace punctuation with space
import string
import re
punct = '|'.join([re.escape(x) for x in string.punctuation])
jobs = jobs.replace(to_replace=punct,value=' ',regex=True)

# Convert to lower case
jobs['description'] = jobs['description'].str.lower()

# Convert description to a list of words
jobs['description'] = jobs['description'].apply(lambda x: x.split())

# Get all bigrams and trigrams for fake and real job postings
from nltk import bigrams, trigrams
all_bigrams_real = [bigram for desc in jobs.description[jobs.fraudulent==0] for bigram in bigrams(desc)]
all_bigrams_fake = [bigram for desc in jobs.description[jobs.fraudulent==1] for bigram in bigrams(desc)]
all_trigrams_real = [trigram for desc in jobs.description[jobs.fraudulent==0] for trigram in trigrams(desc)]
all_trigrams_fake = [trigram for desc in jobs.description[jobs.fraudulent==1] for trigram in trigrams(desc)]

# Get all counts for bigrams and trigrams
def get_all_ngrams_counts(all_ngrams):
    ngram_counts = {}
    for ngram in all_ngrams:
        if ngram in ngram_counts.keys():
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    return ngram_counts


bigram_counts_real = get_all_ngrams_counts(all_bigrams_real)
bigram_counts_fake = get_all_ngrams_counts(all_bigrams_fake)
trigram_counts_real = get_all_ngrams_counts(all_trigrams_real)
trigram_counts_fake = get_all_ngrams_counts(all_trigrams_fake)


# Now let's get the top 10 bigrams
def top_10_ngrams(ngrams_counts):
    i = 1
    for n in sorted(ngrams_counts,key=ngrams_counts.get,reverse=True):
        print("{0}: {1} - {2}".format(i,n,ngrams_counts[n]))
        if i == 10:
            break
        else:
            i+=1

# Now let's get the top 10 bigrams for the cases asked
print("Top Bigrams - Real job postings")
top_10_ngrams(bigram_counts_real)

print("Top Bigrams - Fake job postings")
top_10_ngrams(bigram_counts_fake)

print("Top Trigrams - Real job postings")
top_10_ngrams(trigram_counts_real)

print("Top Trigrams - Fake job postings")
top_10_ngrams(trigram_counts_fake)


# The top 10 for all the cases are dominated by stopwords. Let's remove
# stopwords and re-generate the stats
from gensim.parsing.preprocessing import STOPWORDS
jobs.description = jobs.description.apply(lambda desc: [word for word in desc if word not in STOPWORDS])

all_bigrams_real = [bigram for desc in jobs.description[jobs.fraudulent==0] for bigram in bigrams(desc)]
all_bigrams_fake = [bigram for desc in jobs.description[jobs.fraudulent==1] for bigram in bigrams(desc)]
all_trigrams_real = [trigram for desc in jobs.description[jobs.fraudulent==0] for trigram in trigrams(desc)]
all_trigrams_fake = [trigram for desc in jobs.description[jobs.fraudulent==1] for trigram in trigrams(desc)]

bigram_counts_real = get_all_ngrams_counts(all_bigrams_real)
bigram_counts_fake = get_all_ngrams_counts(all_bigrams_fake)
trigram_counts_real = get_all_ngrams_counts(all_trigrams_real)
trigram_counts_fake = get_all_ngrams_counts(all_trigrams_fake)

print("Top Bigrams - No stopwords - Real job postings")
top_10_ngrams(bigram_counts_real)

print("Top Bigrams - No stopwords - Fake job postings")
top_10_ngrams(bigram_counts_fake)

print("Top Trigrams - No stopwords - Real job postings")
top_10_ngrams(trigram_counts_real)

print("Top Trigrams - No stopwords - Fake job postings")
top_10_ngrams(trigram_counts_fake)
