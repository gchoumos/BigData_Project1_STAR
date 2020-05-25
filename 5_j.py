
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# Read the job descriptions dataset
jobs = pd.read_csv("fake_job_postings.csv",header=0)

########################
# Gaussian Naive Bayes #
########################
gaussian_NB_tc = GaussianNB()

# Split into train and test data. We'll use a split of 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(jobs.telecommuting,jobs.fraudulent,test_size=0.2,random_state=123)
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_pred = gaussian_NB_tc.fit(X_train,y_train).predict(X_test)

correct = (y_test == y_pred).sum()
print("Gaussian NB - Correct predictions {0} out of {1} -- Accuracy: {2:.2f}%"
        .format(correct,X_test.shape[0],correct*100/X_test.shape[0]))

# Calculate precision, recall and f1-score
# Precision = TruePositives / (TruePositives+FalsePositives)
# Recall = TruePositives / (TruePositives+FalseNegatives)
# f1 = A weighted average of precision and recall = 2*(precision*recall)/(precision+recall)
print("Gaussian NB - Precision score: {0}".format(precision_score(y_test,y_pred)))
print("Gaussian NB - Recall score: {0}".format(recall_score(y_test,y_pred)))
print("Gaussian NB - f1 score: {0}".format(f1_score(y_test,y_pred)))


#################
# Decision Tree #
#################
# class_weight='balanced' given the representation mismatch between the 2 classes of the target variable.
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(
        jobs[['telecommuting','has_company_logo','has_questions']],jobs.fraudulent,test_size=0.2,random_state=123)

clf = tree.DecisionTreeClassifier(class_weight='balanced').fit(X_train,y_train)

y_pred = clf.predict(X_test)
correct = (y_test == y_pred).sum()
print("Decision Tree - Correct predictions {0} out of {1} -- Accuracy: {2:.2f}%"
        .format(correct,X_test.shape[0],correct*100/X_test.shape[0]))

# Calculate precision, recall and f1-score for Decision Tree Classifier
print("Decision Tree - Precision score: {0}".format(precision_score(y_test,y_pred)))
print("Decision Tree - Recall score: {0}".format(recall_score(y_test,y_pred)))
print("Decision Tree - f1 score: {0}".format(f1_score(y_test,y_pred)))


############################################################
# Support Vector Machine with 2nd degree polynomial kernel #
############################################################

# No need to recalculate X_train, X_test, y_train and y_test as we are using the
# same features as previously and the plit we've already done fits the purpose.
# Gamma is explicitly set to auto to silence a warning about a change in defaults.
clf = SVC(gamma='auto',class_weight='balanced',kernel='poly',degree=2)

y_pred = clf.fit(X_train,y_train).predict(X_test)

# Print results
correct = (y_test == y_pred).sum()
print("SVM - Correct predictions {0} out of {1} -- Accuracy: {2:.2f}%"
        .format(correct,X_test.shape[0],correct*100/X_test.shape[0]))


# Calculate precision, recall and f1-score for the SVM
print("SVM - Precision score: {0}".format(precision_score(y_test,y_pred)))
print("SVM - Recall score: {0}".format(recall_score(y_test,y_pred)))
print("SVM - f1 score: {0}".format(f1_score(y_test,y_pred)))

####################################################################
####################################################################
####################################################################
# And the logistic regression which was eventually abandoned in order to
# use it on the next sub-question
#######################
# Logistic Regression #
#######################
# class_weight='balanced' given the representation mismatch between the 2 classes of the target variable.
# logreg = LogisticRegression(penalty='l2',tol=1e-4,solver='lbfgs',class_weight='balanced',random_state=123)
# X_train, X_test, y_train, y_test = train_test_split(
#         jobs[['telecommuting','has_company_logo','has_questions']],jobs.fraudulent,test_size=0.2,random_state=123)

# y_pred = logreg.fit(X_train,y_train).predict(X_test)

# Print results
# correct = (y_test == y_pred).sum()
# print("LogReg - Correct predictions {0} out of {1} -- Accuracy: {2:.2f}%"
#         .format(correct,X_test.shape[0],correct*100/X_test.shape[0]))

# Calculate precision, recall and f1-score for logistic regression
# print("LogReg - Precision score: {0}".format(precision_score(y_test,y_pred)))
# print("LogReg - Recall score: {0}".format(recall_score(y_test,y_pred)))
# print("LogReg - f1 score: {0}".format(f1_score(y_test,y_pred)))

