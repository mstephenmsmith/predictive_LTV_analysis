import pandas as pd
import numpy as np
import sys, getopt
import scipy.interpolate
import matplotlib.pyplot as plt
import cPickle as pickle

def LTV(survival_series, margin, discount_rate, freq_='daily'):

	if freq_ == 'daily':
		discount_rate_divider = 365.
	if freq_ == 'weekly':
		discount_rate_divider = 52.
	if freq_ == 'monthly':
		discount_rate_divider = 12.
	if freq_ == 'yearly':
		discount_rate_divider = 1.
	
	sum_series = 0

	for jj, prob in enumerate(survival_series):
		sum_series = sum_series + prob/(1.+discount_rate/float(discount_rate_divider))

	LTV = margin * sum_series

	return LTV

def interpolate_survival(surv_values, num_days):
	y_interp = scipy.interpolate.interp1d(surv_values.index,surv_values.iloc[:,0])

	survival_interp = []

	for ii in xrange(num_days-1):
		survival_interp.append(y_interp(ii+1))

	return survival_interp

def main(argv):

	inputfile = ''
	outputfile = ''

	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print 'test.py -i <inputfile>'
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile>'
			sys.exit()
		else:
			inputfile = arg

	print 'Input file is "', inputfile

	survival_series, buckets, counts_in_bucket, daily_margin = pickle.load(open(inputfile,'rb'))

	LTV_series = []

	num_days = 300
	discount_rate = .15

	for ii, bucket in enumerate(buckets):
		num_days = int(survival_series[ii].index[-1])
		survival_interp = interpolate_survival(survival_series[ii], num_days)
		LTV_temp = LTV(survival_interp, daily_margin[ii], discount_rate)
		LTV_series.append(LTV_temp)

	for ii in xrange(len(buckets)):
		print "LTV for bucket "+buckets[ii]+" is ", LTV_series[ii]

	pos = np.arange(len(buckets))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(buckets)

	plt.title('LTVs Number of Uses Cohorts')
	plt.xlabel('Number of Uses Cohorts')
	plt.ylabel('LTV ($)')

	plt.bar(pos, LTV_series, width)

	plt.savefig("LTV.png")

if __name__ == '__main__':
	main(sys.argv[1:])