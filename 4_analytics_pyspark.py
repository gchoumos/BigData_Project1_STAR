
# TODO - Exchange rates should be calculated programmatically through
# a library. Maybe something along the lines of yahoo finance?
# Same applies if possible for the country's currency.
import pyspark.sql.functions as f
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import IntegerType, StringType, FloatType


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


# Though the previous may lead us to think that we have a couple of easy cases to drop, things
# change if we also take into account location.
jobs.select('min_salary','max_salary','location','fraudulent').orderBy('min_salary',ascending=False).show(5)

# We can that the huge salaries are for India and given they are *not* fraudulent, we realize that the case is
# we have salaries in currencies specific to the country, which makes sense. We'll proceed with some significant
# preprocessing to that data as a result.

# We are interested in the country so that we convert the salary currency accordingly.
# Define a function that will extract the Country from Location.
def getCountry(value):
    return value.split(',')[0]

# Convert it to a udf and add the new country column to the dataset
udf_getCountry = f.udf(getCountry, StringType())
jobs = jobs.withColumn("country",udf_getCountry("location"))

# There are 26 distinct countries
jobs.select('country').distinct().count()

# Let's have a look at which they are
jobs.select('country').distinct().show(26)

# Country to Currency mapping
currency = {
    'AU': 'AUD', # Australian Dollar
    'BR': 'BRL', # Brazilian Real
    'CA': 'CAD', # Canadian Dollar
    'DE': 'EUR', # Euro
    'EG': 'EGP', # Egyptian Pound
    'FR': 'EUR', # Euro
    'GB': 'GBP', # Pound Sterling
    'GR': 'EUR', # Euro
    'HK': 'HKD', # Hong Kong Dollar
    'ID': 'IDR', # Indonesian Rupiah
    'IN': 'INR', # Indian Rupee
    'IQ': 'IQD', # Iraqi Dinar
    'IT': 'EUR', # Euro
    'KH': 'KHR', # Cambodian Riel
    'MU': 'MUR', # Mauritian Rupee
    'MY': 'MYR', # Malaysian Ringgit
    'NL': 'EUR', # Euro
    'NZ': 'NZD', # New Zealand Dollar
    'PH': 'PHP', # Philippine Peso
    'PK': 'PKR', # Pakistani Rupee
    'SA': 'ZAR', # South African Rand
    'SG': 'SGD', # Singapore Dollar
    'TH': 'THB', # Thai Baht
    'TR': 'TRY', # Turkish Lira
    'TW': 'TWD', # New Taiwan Dollar
    'US': 'USD', # United States Dollar
}

# Any Currency to USD - The value it has to be multiplied with to be expressed in USD
# Rates as of May 6, 2020
# These rates could be obtained through various means, like for example through the
# Yahoo Finance libraries.
exchange_rate_USD = {
    'AUD': 0.64,
    'BRL': 0.18,
    'CAD': 0.71,
    'EGP': 0.064,
    'EUR': 1.08,
    'GBP': 1.24,
    'HKD': 0.13,
    'IDR': 0.000067,
    'INR': 0.013,
    'IQD': 0.00084,
    'KHR': 0.00024,
    'MUR': 0.025,
    'MYR': 0.23,
    'NZD': 0.61,
    'PHP': 0.020,
    'PKR': 0.0063,
    'ZAR': 0.054,
    'SGD': 0.71,
    'THB': 0.031,
    'TRY': 0.14,
    'TWD': 0.034,
    'USD': 1.0,
}

# Minimum salary of a country expressed in USD
# Data as of May 6 - 2020 -- https://en.wikipedia.org/wiki/List_of_minimum_wages_by_country
minimum_salary_USD = {
    'AU': 28768,
    'BR': 3500,
    'CA': 18112,
    'DE': 22525,
    'EG': 816, # 68 per month for the public sector - approximation
    'FR': 21142,
    'GB': 24183, # differs by age - approximation
    'GR': 10731,
    'HK': 10049,
    'ID': 1304,
    'IN': 767,
    'IQ': 2534,
    'IT': 11000, # approximation - No minimum salary in Italy, mostly collective agreements.
    'KH': 2280, # approximation
    'MU': 1513, # approximation
    'MY': 2566,
    'NL': 22876,
    'NZ': 27881,
    'PH': 2770,
    'PK': 1991,
    'SA': 3511,
    'SG': 9000, # approximation - no minimum wage
    'TH': 3034,
    'TR': 9676,
    'TW': 9088,
    'US': 15080,
}


# We need a new udf for the conversion
def toUSD(country,value):
    # Casting to an integer as the occuring decimals are of no significance
    return int(value*exchange_rate_USD[currency[country]])

udf_toUSD = f.udf(toUSD, IntegerType())

# And create the new columns for min and max salary in USD
jobs = jobs.withColumn("min_salary_USD",udf_toUSD("country","min_salary"))
jobs = jobs.withColumn("max_salary_USD",udf_toUSD("country","max_salary"))


# We are not done yet. We should also consider the employment type as not all jobs are full-time.
# Here's what we have in the remaining data grouped by employment type and fraudulent/non-fraudulent
jobs.groupBy('employment_type','fraudulent').count().show()

# And let's check how the converted min_salary_USD and max_salary_USD compare on average
# for each of the employment types
jobs.groupBy('employment_type').agg({'min_salary_USD': 'mean','max_salary_USD': 'mean'}).show()

#
# The decision is to hold fire for now with preprocessing in order to move on.
#

# Now that we have the converted salaries, it is time to identify and clean up some outliers. First of all,
# let's have a look at how many rows exist with maximum salary, or minimum salary or even both being 0.
jobs.select('min_salary_USD','max_salary_USD').filter('min_salary_USD = 0 or max_salary_USD = 0').count()

# And just being curious, let's also see what are the distinct cases out of them
jobs.select('min_salary_USD','max_salary_USD').filter('min_salary_USD = 0 or max_salary_USD = 0').distinct().show()

# Given they are not useful and can lead us to erroneous conclusions, we are dropping them.
jobs = jobs.filter('min_salary_USD > 0 and max_salary_USD > 0')
jobs.count()

# Before moving on, a quick sanity check that all values of min_salary are smaller or equal to their max_salary counterpart
jobs.filter('min_salary <= max_salary').count()

# To be able to compare salaries in a meaningful way, we'll create columns based on how they relate to the country's
# minimum wage expressed in USD. So for a min_salary of 25000 in a country with the minimum is 10000, the factor will be 2.5
# We need a new UDF for this
def toSalaryFactor(country,salary):
    # Return how the salary in USD compares to the minimum salary of the country (again in USD)
    return float(salary/minimum_salary_USD[country])

udf_toSalaryFactor = f.udf(toSalaryFactor, FloatType())

# Create the new columns for min and max salary factors
jobs = jobs.withColumn("min_salary_factor",udf_toSalaryFactor("country","min_salary_USD"))
jobs = jobs.withColumn("max_salary_factor",udf_toSalaryFactor("country","max_salary_USD"))

# Convert to Pandas dataframe
pd_jobs = jobs.toPandas()

# Let's check a histogram of the min_salary_factor and max_salary_factor
plt.hist(pd_jobs.min_salary_factor.values,bins=100,alpha=0.5,label='min salary factor')
plt.hist(pd_jobs.max_salary_factor.values,bins=100,alpha=0.5,label='max salary factor')
plt.xlabel('min and max salary factor',fontsize=20)
plt.ylabel('Occurences',fontsize=20)
plt.legend(loc='upper right',fontsize=15)
plt.show()

# The above histogram was both too crowded on the left, and way too spread out to the right
# So it looks like a good idea to plot in a logarithmic scale for the min and max salary factors
plt.hist(np.log10(pd_jobs.min_salary_factor.values),bins=100,alpha=0.5,label='min salary')
plt.hist(np.log10(pd_jobs.max_salary_factor.values),bins=100,alpha=0.5,label='max salary')
plt.xlabel('Logarithm of min and max salary',fontsize=20)
plt.ylabel('Occurences',fontsize=15)
plt.legend(loc='upper left',fontsize=15)
plt.show()

# Before dropping the outliers, let's see how many of those will be affected are fraudulent
jobs.filter((f.col('min_salary_factor')<0.4) | (f.col('max_salary_factor')>15.8)).groupBy('fraudulent').count().show()

# Drop the outliers - Any with min_salary_factor < 0.4 or max_salary_factor > 15.8
jobs = jobs.filter((f.col('min_salary_factor')>0.4) & (f.col('max_salary_factor')<15.8))

# Recompute average and standard deviation of maximum salary - Fake job postings
# Note that we are now using the new feature, max_salary_factor, for the statistics!
jobs.filter("fraudulent == 1").agg({'max_salary_factor': 'mean'}).show()
jobs.filter("fraudulent == 1").agg({'max_salary_factor': 'stddev'}).show()

# Re-Compute the median. After removing the outliers we now have 612 real job postings. So
# the median will be Thus, the median will be the average of rows 306 and 307. Note that we
# actually use indexes 305 and and 306 because indexing in Python starts at 0.
salary1 = jobs.filter("fraudulent == 0").select("min_salary_factor").sort("min_salary_factor").collect()[305][0]
salary2 = jobs.filter("fraudulent == 0").select("min_salary_factor").sort("min_salary_factor").collect()[306][0]
print("Median of min salary for real job postings is {0}".format((salary1+salary2)/2))

############################
# Moving to sub-question f #
############################

# It's the last question of the section and we only need the description and fraudulent.
# Let's keep only these 2 columns and also convert them to a Pandas dataframe.
jobs = jobs.select('description','fraudulent').toPandas()

# Clean up a bit by replacing the following with a space. The first 3 are just masked values
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
