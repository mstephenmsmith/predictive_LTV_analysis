import pandas as pd
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt
import cPickle as pickle

def plot_survival_rates(kmf_values, unique_buckets, sav_name):

	for jj, surv in enumerate(kmf_values):
		if jj==0:
			ax = plt.plot(surv.index,surv.iloc[:,0], label = unique_buckets[jj])
			plt.title('Survival Function for Cohorts')
			plt.xlabel('Days of Survival')
			plt.ylabel('Probability')

		else:
			plt.plot(surv.index,surv.iloc[:,0], label = unique_buckets[jj])

	plt.legend()

	plt.savefig(sav_name)

	plt.savefig("survival_rates.png")

	plt.clf()

def plot_use_count_hist(unique_buckets, counts_in_bucket, sav_name):

	pos = np.arange(len(unique_buckets))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(unique_buckets)

	plt.title('Counts for Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('Count')

	plt.bar(pos, counts_in_bucket, width)

	plt.savefig(sav_name)

	plt.clf()

def plot_LTV_hist(LTV_series, unique_buckets, sav_name):

	pos = np.arange(len(unique_buckets))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(unique_buckets)

	plt.title('LTVs Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('LTV ($)')

	plt.bar(pos, LTV_series, width)

	plt.savefig(sav_name)

	plt.clf()

def main(inputfile, outputfile=None):
	
	LTV_series, kmf_values, unique_buckets, counts_in_bucket, daily_margin = pickle.load(open(inputfile,'rb'))

	plot_survival_rates(kmf_values, unique_buckets, "survival_rates.png")

	plot_use_count_hist(unique_buckets, counts_in_bucket, "use_count_hist.png")

	plot_LTV_hist(LTV_series, unique_buckets, "LTV.png")


if __name__ == '__main__':
	main(inputfile)