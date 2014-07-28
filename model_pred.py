from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import svm
from sklearn.cross_validation import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, f1_score, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix


def get_feature_matrix_labels(df_main, df_att, LTV_split):

	df_main['first_purch_date'] = pd.to_datetime(df_main['first_purch_date'])
	
	df_main['first_use_date'] = pd.to_datetime(df_main['first_use_date'])

	df_main['first_use_to_first_purch'] = df_main['first_purch_date'] - df_main['first_use_date']

	df_main = df_main.iloc[np.where(df_main['first_use_to_first_purch']>0.)[0]]

	df_main['first_use_to_first_purch'] = df_main['first_use_to_first_purch'].apply(lambda x: x.item()/float(8.64*10**13))

	df = pd.merge(df_main, df_att[['user_id','user_source']], on = 'user_id', how = 'left')

	store_dummies = pd.core.reshape.get_dummies(df['most_used_store'])

	source_dummies = pd.core.reshape.get_dummies(df['user_source'])

	X = df[['first_use_to_first_purch','mean_freq','std_freq','num_items_purch','first_purchase_amount']]
	#X = df[['first_use_to_first_purch','mean_freq','std_freq','first_purchase_amount']]

	#X = pd.concat([X, source, store_dummies], axis = 1)

	labels = np.where(df.LTV<LTV_split,1,0)

	return X, labels


def run_model(Model, X_train, X_test, y_train, y_test):
	if Model == LR:
		m = Model(C=0.01)
	elif Model == svm:
		m = Model.SVC(kernel='rbf', probability=True)
	else:
		m = Model()
	fit_ = m.fit(X_train, y_train)
	y_predict = m.predict(X_test)
	probas_ = fit_.predict_proba(X_test)
	fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
	return accuracy_score(y_test, y_predict), \
		f1_score(y_test, y_predict), \
		precision_score(y_test, y_predict), \
		recall_score(y_test, y_predict), \
		roc_auc_score(y_test, probas_[:,1]), fpr, tpr, confusion_matrix(y_test, y_predict)
		# auc(fpr,tpr), fpr, tpr, confusion_matrix(y_test, y_predict)

def get_training_test_indices(labels, num_folds):

	kf = KFold(labels.shape[0], n_folds=num_folds)

	return kf

def get_scores(X, labels, models, kf):

	scores_ = []
	
	for name, Model in models.iteritems():

		accs_ = []
		f1s_ = []
		precs_ = []
		recs_ = []
		roc_aucs_ = []

		for train_index, test_index in kf:

			X_train, X_test = X.iloc[train_index], X.iloc[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			acc, f1, prec, rec, roc_auc, fpr, tpr, confusion_matrix_ = run_model(Model, X_train, X_test, y_train, y_test)
			# print "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, roc_auc, name)

			accs_.append(acc)
			f1s_.append(f1)
			precs_.append(prec)
			recs_.append(rec)
			roc_aucs_.append(roc_auc)
			roc_aucs_.append(roc_auc)

		# print "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % (np.mean(accs_), np.mean(f1s_), np.mean(precs_), np.mean(recs_), np.mean(roc_aucs_), name)

		scores_.append([name, np.mean(accs_), np.mean(f1s_), np.mean(precs_), np.mean(recs_), np.mean(roc_aucs_)])

	return scores_

def get_fprs_tprs(X, labels, models):

	# this function is mainly used to get fprs and tprs to be used for later plotting

	X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

	fprs_ = []
	tprs_ = []

	for name, Model in models.iteritems():
		acc, f1, prec, rec, roc_auc, fpr, tpr, confusion_matrix_ = run_model(Model, X_train, X_test, y_train, y_test)	
		fprs_.append(fpr)
		tprs_.append(tpr)

	return fprs_, tprs_


def main(inputfile_feat, inputfile_attribute, outputfile, LTV_split = 200):
	
	df_main = pd.read_csv(inputfile_feat)

	df_att = pd.read_csv(inputfile_attribute)

	X, labels = get_feature_matrix_labels(df_main, df_att, LTV_split)

	print "Split of labels: ", Counter(labels)

	models = {"Logistic Regression": LR, \
		  "kNN": KNeighborsClassifier, \
		  "Naive Bayes": MultinomialNB, \
		  "Random Forest": RF }#, \
		  # "SVM": svm}

	num_folds = 3

	kf = get_training_test_indices(labels, num_folds)

	scores_ = get_scores(X, labels, models, kf)

	fprs_, tprs_ = get_fprs_tprs(X, labels, models)

	pickle.dump((scores_, fprs_, tprs_), open(outputfile, 'wb'))

if __name__ == '__main__':
	main(inputfile_feat, inputfile_attribute, outputfile)
