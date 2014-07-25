from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import auc, f1_score, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix

def run_model(Model, X_train, X_test, y_train, y_test):
	if Model == LR:
		m = Model(C=0.1)
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
		   roc_auc_score(y_test, y_predict), fpr, tpr, confusion_matrix(y_test, y_predict)
		   # auc(fpr,tpr), fpr, tpr, confusion_matrix(y_test, y_predict)

def main():
	df = pd.read_csv('/Users/mstephenmsmith/Zipfian/capstone_project/data/final_feature_matrix.csv')

	df_att = pd.read_csv('/Users/mstephenmsmith/Zipfian/capstone_project/data/user_attributes.csv')

	df = pd.merge(df, df_att[['user_id','user_source']], on = 'user_id', how = 'left')

	store_dummies = pd.core.reshape.get_dummies(df['most_used_store'])

	source_dummies = pd.core.reshape.get_dummies(df['user_source'])

	# X = df[['mean_freq','num_items_purch','first_purchase_amount']]

	X = df[['mean_freq','std_freq','first_purchase_amount']]

	X = pd.concat([X, store_dummies, source_dummies], axis = 1)

	#label = np.where(df.LTV<np.mean(df.LTV),1,0)
	label = np.where(df.LTV<300.,1,0)

	print Counter(label)

	X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.33)

	# clf = LR()

	# clf.fit(X_train,y_train)

	# print "Score: ",clf.score(X_test,y_test)

	models = {"Logistic Regression": LR, \
		  "kNN": KNeighborsClassifier, \
		  "Naive Bayes": MultinomialNB, \
		  "Random Forst": RF}


	# fig, (ax0, ax1)  = plt.subplots(1,2)

	prec_rec_curve = []
	for_roc_curve = []

	for name, Model in models.iteritems():
		acc, f1, prec, rec, roc_auc, fpr, tpr, confusion_matrix_ = run_model(Model, X_train, X_test, y_train, y_test)
		print "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, roc_auc, name)
		prec_rec_curve.append([prec,rec])
		for_roc_curve.append([fpr,tpr])
		plt.plot(fpr, tpr, label=name+' (area = %0.2f)' % roc_auc)
		# ax1.plot(prec,rec, label=name)

	# ax0.plot(np.transpose(for_roc_curve))
	# ax1.plot(prec_roc_curve)


	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

