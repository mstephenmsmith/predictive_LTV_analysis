import pandas as pd
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt
import cPickle as pickle

# pd.set_option('display.mpl_style', 'default')

def plot_survival_rates(kmf_values, bucket_names, sav_name):

	fig = plt.figure(figsize=(7,5))

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

	plt.savefig(sav_name, dpi=500)

	plt.clf()

def plot_use_count_hist(bucket_names, counts_in_bucket, sav_name):

	fig = plt.figure(figsize=(7,5))
	pos = np.arange(len(bucket_names))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(bucket_names)

	plt.title('Counts for Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('Count')

	plt.bar(pos, counts_in_bucket, width)

	plt.savefig(sav_name, dpi=500)

	plt.clf()

def plot_LTV_hist(LTV_series, LTV_bucket_vals, bucket_names, bucket_boundaries, sav_name):

	fig = plt.figure(figsize=(7,5))
	pos = np.arange(len(bucket_names))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(bucket_names)

	plt.title('LTVs Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('Mean LTV ($)')

	plt.bar(pos, LTV_bucket_vals, width)

	plt.savefig(sav_name, dpi=500)

	plt.clf()

def plot_LTV_scatter(LTV_bucket_vals, bucket_medians, counts_in_bucket, sav_name):

	fig = plt.figure(figsize=(7,5))

	plt.title('LTVs Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('Mean LTV ($)')

	plt.scatter(bucket_medians, counts_in_bucket, s = LTV_bucket_vals)

	plt.savefig(sav_name, dpi=500)

	plt.clf()


def plot_roc_curves(scores_list, fprs_, tprs_, sav_name):

	models_to_plot = ['Logistic Regression', 'Random Forest']

	fig = plt.figure(figsize=(7,5))
	ax = fig.add_subplot(111)
	ax.set_aspect('equal')

	for ii, model in enumerate(scores_list):
		name = model[0]
		if [x for x in models_to_plot if name in models_to_plot]:
			fpr = fprs_[ii]
			tpr = tprs_[ii]
			name = model[0]
			roc_auc = model[5]
			plt.plot(fpr,tpr, linewidth = 2, label=name+' (area = %0.2f)' % roc_auc)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])


	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right", prop={'size':12})
	plt.savefig(sav_name, dpi=500)

def main(inputfile_LTV, inputfile_model, bucket_boundaries, folder_to_save, outputfile=None):
	
	LTV_series, kmf_values, LTV_bucket_vals, bucket_names, bucket_medians, counts_in_bucket, daily_margin = pickle.load(open(inputfile_LTV,'rb'))

	scores_list, tprs_, fprs_ = pickle.load(open(inputfile_model,'rb'))

	plot_survival_rates(kmf_values, bucket_names, folder_to_save + "survival_rates.png")

	plot_use_count_hist(bucket_names, counts_in_bucket, folder_to_save + "use_count_hist.png")

	plot_LTV_hist(LTV_series, LTV_bucket_vals, bucket_names, bucket_boundaries, folder_to_save + "LTV.png")

	plot_LTV_scatter(LTV_bucket_vals, bucket_medians, counts_in_bucket, folder_to_save + "LTV_scatter.png")

	plot_roc_curves(scores_list, tprs_, fprs_, folder_to_save + "roc_curves.png")

if __name__ == '__main__':
	main(inputfile_LTV, inputfile_model)