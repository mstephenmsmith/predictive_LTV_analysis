import pandas as pd
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt
import cPickle as pickle

def plot_survival_rates(kmf_values, bucket_names, sav_name):

	for jj, surv in enumerate(kmf_values):
		if jj==0:
			ax = plt.plot(surv.index,surv.iloc[:,0], label = bucket_names[jj])
			plt.title('Survival Function for Cohorts')
			plt.xlabel('Days of Survival')
			plt.ylabel('Probability')

		else:
			plt.plot(surv.index,surv.iloc[:,0], label = bucket_names[jj])

	plt.legend()

	plt.savefig(sav_name)

	plt.savefig("survival_rates.png")

	plt.clf()

def plot_use_count_hist(bucket_names, counts_in_bucket, sav_name):

	pos = np.arange(len(bucket_names))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(bucket_names)

	plt.title('Counts for Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('Count')

	plt.bar(pos, counts_in_bucket, width)

	plt.savefig(sav_name)

	plt.clf()

def plot_LTV_hist(LTV_series, bucket_names, bucket_vals, sav_name):

	pos = np.arange(len(bucket_names))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(bucket_names)

	plt.title('LTVs Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('LTV ($)')

	plt.bar(pos, np.histogram(LTV_series,bins=bucket_vals)[0], width)

	plt.savefig(sav_name)

	plt.clf()

def plot_roc_curves(scores_list, fprs_, tprs_, sav_name):

	for ii, model in enumerate(scores_list):
		fpr = fprs_[ii]
		tpr = tprs_[ii]
		name = model[0]
		roc_auc = model[5]
		plt.plot(fpr,tpr, label=name+' (area = %0.2f)' % roc_auc)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig(sav_name)

def main(inputfile_LTV, inputfile_model, bucket_vals, folder_to_save, outputfile=None):
	
	LTV_series, kmf_values, bucket_names, counts_in_bucket, daily_margin = pickle.load(open(inputfile_LTV,'rb'))

	scores_list, tprs_, fprs_ = pickle.load(open(inputfile_model,'rb'))

	plot_survival_rates(kmf_values, bucket_names, folder_to_save + "survival_rates.png")

	plot_use_count_hist(bucket_names, counts_in_bucket, folder_to_save + "use_count_hist.png")

	plot_LTV_hist(LTV_series, bucket_names, bucket_vals, folder_to_save + "LTV.png")

	plot_roc_curves(scores_list, tprs_, fprs_, folder_to_save + 'roc_curves.png')

if __name__ == '__main__':
	main(inputfile_LTV, inputfile_model)