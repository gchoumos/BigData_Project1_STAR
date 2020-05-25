#!/usr/bin/python3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import string
import warnings
import re

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')

# Read the jobs dataset
jobs = pd.read_csv("fake_job_postings.csv",header=0)

# A function that we will use to preprocess text fields
def preprocess_text(jobs,colname='description'):
    print("###################################"+"#"*len(colname))
    print("# Text preprocessing for column: {0} #".format(colname))
    print("###################################"+"#"*len(colname))
    print("Removing URLs, EMAILs and PHONEs...")
    jobs[[colname]] = jobs[[colname]].replace([r'#(URL|EMAIL|PHONE)_\S*#'],' ', regex=True)
    print("Removing the '\\xa0' special character...")
    jobs[[colname]] = jobs[[colname]].replace([r'\xa0',r'&\w+;'],' ', regex=True)
    print("Replacing punctuation with space...")
    punct = '|'.join([re.escape(x) for x in string.punctuation])
    jobs = jobs.replace(to_replace=punct,value=' ',regex=True)
    print("Converting to lower case...")
    jobs[colname] = jobs[colname].str.lower()
    ####
    ## Stopwords are being removed as part of the pipeline step: CountVectorizer
    ####
    # print("Removing digit-only words...")
    # jobs[[colname]] = jobs[[colname]].replace([r'\b[0-9]+\b'],' ',regex=True)
    print("Removing words of length 1 (occured after punctuation replacement)...")
    jobs[[colname]] = jobs[[colname]].replace([r'\b\w\b'],' ',regex=True)
    return jobs[[colname]]

##########################
##########################
## Use description only ##
##########################
##########################
jobs.description = preprocess_text(jobs,colname='description')

# To convert the description to tf-idf representation we use the CountVectorizer and
# TfIdfTransformer. We use the Feature Union in order to consider the unigrams and
# bi-grams as different features and be able to weight their contribution separately.
# The pipeline is there to bind everything together.
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from helpers import ItemSelector, currency, exchange_rate_USD, minimum_salary_USD

desc_logreg = LogisticRegression(
    penalty='l2',
    tol=1e-4,
    solver='lbfgs',
    max_iter=300,
    class_weight='balanced')

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            # Description unigrams
            ('desc_unigrams', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_desc_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Comment bigrams
            ('desc_bigrams', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(2,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_desc_bi', SelectFromModel(desc_logreg,threshold=None)),
            ])),

        ],

        # Weight components in FeatureUnion - Here are the optimals
        transformer_weights={
            'desc_unigrams': 1.20,
            'desc_bigrams':  1.00,
            # 'desc_trigrams': 1.00, # Trigrams didn't make this any better
        },
    )),

    ('logr', LogisticRegression(penalty='l2', tol=0.0001, class_weight='balanced')),
])

# We have already selected the parameters for which the GridSearch provided the best results.
# Just leaving this here for visibility. Note that for more than one values, separate fits will
# be run to find out the best combination of parameters.
parameters = {
    'union__desc_unigrams__vect__min_df': [0.00001],
    'union__desc_bigrams__vect__min_df': [0.0001],
    'union__desc_unigrams__vect__max_df': [0.4],
    'union__desc_bigrams__vect__max_df': [0.6],
}
accuracy_scorer = make_scorer(accuracy_score,greater_is_better=True)
precision_scorer = make_scorer(precision_score,greater_is_better=True)
recall_scorer = make_scorer(recall_score,greater_is_better=True)
f1_scorer = make_scorer(f1_score,greater_is_better=True)


scorers = {
    'accuracy': accuracy_scorer,
    'precision': precision_scorer,
    'recall': recall_scorer,
    'f1': f1_scorer
}

grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=1, scoring=scorers, refit='f1')
grid_search.fit(jobs,jobs.fraudulent)

# Display the (best) score of the model and parameters.
print("Best parameters to optimise f1 score: {0}".format(grid_search.best_params_))
print("####################################################")
print("# SCORES - Description only - Unigrams and Bigrams #")
print("####################################################")
print("f1: {0:.3f}%".format(grid_search.best_score_*100))
print("accuracy: {0:.3f}%".format(grid_search.cv_results_['mean_test_accuracy'][0]*100))
print("precision: {0:.3f}%".format(grid_search.cv_results_['mean_test_precision'][0]*100))
print("recall: {0:.3f}%".format(grid_search.cv_results_['mean_test_recall'][0]*100))



###
########
#############
##################
#######################
############################
#################################
######################################
######################################
## Use rest of the features as well ##
######################################
######################################
#################################
############################
#######################
##################
#############
########
###

# Add the company profile. We will preprocess it like the way we did for "description"
jobs.company_profile = preprocess_text(jobs,colname='company_profile')
jobs.requirements = preprocess_text(jobs,colname='requirements')
jobs.benefits = preprocess_text(jobs,colname='benefits')
jobs.title = preprocess_text(jobs,colname='title')

# I'll combine "department", "function" and "industry" columns into 1 - "dep_ind_fun"
# To do that, I'll replace the NaNs with the empty string before concatenating them.
jobs.department = jobs.department.fillna('')
jobs.industry = jobs.industry.fillna('')
jobs.function = jobs.function.fillna('')
jobs['dep_ind_fun'] = jobs.department + ' ' + jobs.industry + ' ' + jobs.department
# And then preprocess it like the rest of the text features
jobs.dep_ind_fun = preprocess_text(jobs,colname='dep_ind_fun')

# Let's do the same for location and see how it goes.
# UPDATE: Abandoned because it didn't contibute
jobs.location = jobs.location.fillna('')
# jobs.location = preprocess_text(jobs,colname='location')

# Combine required exprerience and required education and use them as 1 new feature
# UPDATE: Abandoned - Not only it didn't contribute but actually made things worse
jobs.required_experience = jobs.required_experience.fillna('')
jobs.required_education = jobs.required_education.fillna('')
jobs['exp_edu'] = jobs.required_experience + ' ' + jobs.required_education
jobs.exp_edu = preprocess_text(jobs,colname='exp_edu')

# Employment Type
jobs.employment_type = jobs.employment_type.fillna('')
jobs.employment_type = preprocess_text(jobs,colname='employment_type')

# Salary Range - This is going to be handled like what was done in question 4.
# Steps:
# - Separate min and max salary and keep in new columns
# - Consider the bad values we have for salary range like Oct-15 (turn them to 0-0)
# - Take into account location (country)
# - Take into account currency of that country
# - Use USD as the same base
# - Find exchange rates of the local currency to USD
# - Convert all to USD
# - Find minimum salaries of each country (actual or approximations)
# - Create new columns: min_salary_factor and max_salary_factor which will
#   express the salaries as a factor of that country's minimum 
# - Use the new columns as features
# ---------------
jobs.salary_range = jobs.salary_range.fillna('0-0')
jobs.salary_range = jobs.salary_range.str.split('-')
jobs.salary_range = jobs.salary_range.apply(lambda x: x if len(x)==2 else ['0','0'])
# This is to take care of cases of salary ranges like 15-Oct
jobs.salary_range = jobs.salary_range.apply(lambda x: x if x[0].isnumeric() and x[1].isnumeric() else ['0','0'])
jobs.location = jobs.location.fillna(',,')
jobs.location = jobs.location.str.split(',')
jobs['country'] = jobs.location.apply(lambda x: x[0])
jobs['min_salary'] = jobs.salary_range.apply(lambda x: x[0] if len(x) > 0 else '0')
jobs['max_salary'] = jobs.salary_range.apply(lambda x: x[1] if len(x) > 1 else '0')
jobs.min_salary = jobs.min_salary.astype(int)
jobs.max_salary = jobs.max_salary.astype(int)

# Min and Max salary to USD
jobs['min_salary_USD'] = jobs.apply(
    lambda x:
    x['min_salary']*exchange_rate_USD[currency[x['country']]]
    if x['min_salary']!=0 and x['min_salary'] != '0' and x['country'] != '' and len(x['country'])==2
    else 0, axis=1)

jobs['max_salary_USD'] = jobs.apply(
    lambda x:
    x['max_salary']*exchange_rate_USD[currency[x['country']]]
    if x['min_salary']!=0 and x['max_salary'] != '0' and x['country'] != '' and len(x['country'])==2
    else 0, axis=1)

# Min and Max salary factors
jobs['min_salary_factor'] = jobs.apply(
    lambda x:
    float(x['min_salary_USD']/minimum_salary_USD[x['country']])
    if x['min_salary_USD'] !=0 and x['max_salary_USD'] != 0 and x['country'] != '' and len(x['country'])==2
    else 0, axis=1)

jobs['max_salary_factor'] = jobs.apply(
    lambda x:
    float(x['max_salary_USD']/minimum_salary_USD[x['country']])
    if x['min_salary_USD'] !=0 and x['max_salary_USD'] != 0 and x['country'] != '' and len(x['country'])==2
    else 0, axis=1)


# Try length of benefits as suggested
jobs.benefits = jobs.benefits.fillna('')
jobs['benefits_len'] = jobs.benefits.apply(lambda x: len(x))

# Telecommuting - No NaNs
# Has Company Logo - No NaNs
# Has Questions = No NaNs

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            # Description unigrams
            ('desc_unigrams', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_desc_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Description bigrams
            ('desc_bigrams', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(2,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_desc_bi', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Company Profile unigrams
            ('comp_prof_unigrams', Pipeline([
                ('selector', ItemSelector(key='company_profile')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_comp_prof_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Company Profile bigrams
            ('comp_prof_bigrams', Pipeline([
                ('selector', ItemSelector(key='company_profile')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(2,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_comp_prof_bi', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Requirements unigrams
            ('req_unigrams', Pipeline([
                ('selector', ItemSelector(key='requirements')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_req_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Benefits unigrams
            ('ben_unigrams', Pipeline([
                ('selector', ItemSelector(key='benefits')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_ben_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Title unigrams
            ('tit_unigrams', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_tit_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Department, Industry and Function combined feature - Unigrams
            ('dep_ind_fun_unigrams', Pipeline([
                ('selector', ItemSelector(key='dep_ind_fun')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_dep_ind_fun_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Employment type - Unigrams
            ('empl_unigrams', Pipeline([
                ('selector', ItemSelector(key='employment_type')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english',ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_empl_uni', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Telecommuting
            ('telecommuting', Pipeline([
                ('selector', ItemSelector(key='telecommuting')),
                ('sfm_tel', SelectFromModel(desc_logreg,threshold=None)),
            ])),

            # Length of benefits
            ('benefits_len', Pipeline([
                ('selector', ItemSelector(key='benefits_len')),
                ('sfm_benlen', SelectFromModel(desc_logreg,threshold=None)),
            ])),

        ],

        # Weight components in FeatureUnion - Here are the optimals
        transformer_weights={
            'desc_unigrams': 1.6,
            'desc_bigrams':  0.9,
            'comp_prof_unigrams': 1.30,
            'comp_prof_bigrams': 1.00,
            'req_unigrams': 0.8,
            'ben_unigrams': 0.9,
            'tit_unigrams': 1.2,
            'dep_ind_fun_unigrams': 0.60,
            'empl_unigrams': 1.0,
            'telecommuting': 0.7,
            # 'benefits_len': 0.8, # length of benefits - abandoned
            # 'min_sf': 0.6, # min_salary_factor - Didn't really contribute - abandoned
            # 'max_sf': 0.6, # max_salary_factor - Didn't really contribute - abandoned
            # 'questions': 0.7, # has_questions - Didn't contribute, so it was abandoned
            # 'logo': 0.6, # has_company_logo - Didn't contribute, so it was abandoned
            # 'exp_edu': 0.80, # Required experience and required education made things worse
            # 'loc_unigrams': 0.5, # Location didn't contribute, so it was abandoned
            # 'desc_trigrams': 1.00, # Trigrams didn't make this any better
        },
    )),

    ('logr', LogisticRegression(penalty='l2',max_iter=300, tol=0.001,class_weight='balanced')),
])


# Have already tested some values and the following constitute a good set
parameters = {
    'union__desc_unigrams__vect__min_df': [0.00001],
    'union__desc_bigrams__vect__min_df': [0.0001],
    'union__desc_unigrams__vect__max_df': [0.4],
    'union__desc_bigrams__vect__max_df': [0.6],
    'union__comp_prof_unigrams__vect__min_df': [0.00001],
    'union__comp_prof_bigrams__vect__min_df': [0.0001],
    'union__comp_prof_unigrams__vect__max_df': [0.4],
    'union__comp_prof_bigrams__vect__max_df': [0.6],
    'union__req_unigrams__vect__min_df': [0.00001],
    'union__req_unigrams__vect__max_df': [0.4],
    'union__ben_unigrams__vect__min_df': [0.00001],
    'union__ben_unigrams__vect__max_df': [0.5],
    'union__tit_unigrams__vect__min_df': [0.00001],
    'union__tit_unigrams__vect__max_df': [0.5],
    'union__dep_ind_fun_unigrams__vect__min_df': [0.00001],
    'union__dep_ind_fun_unigrams__vect__max_df': [0.5],
    'union__empl_unigrams__vect__min_df': [0.00001],
    'union__empl_unigrams__vect__max_df': [0.5],
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring=scorers, refit='f1')
grid_search.fit(jobs,jobs.fraudulent)

# Display the (best) score of the model and parameters.
# print("Best parameters to optimise f1 score: {0}".format(grid_search.best_params_))
print("#########################################################")
print("# SCORES - All selected features (original and derived) #")
print("#########################################################")
print("f1: {0:.3f}%".format(grid_search.best_score_*100))
print("accuracy: {0:.3f}%".format(grid_search.cv_results_['mean_test_accuracy'][0]*100))
print("precision: {0:.3f}%".format(grid_search.cv_results_['mean_test_precision'][0]*100))
print("recall: {0:.3f}%".format(grid_search.cv_results_['mean_test_recall'][0]*100))

import pdb
pdb.set_trace()
######################
######################
######################

