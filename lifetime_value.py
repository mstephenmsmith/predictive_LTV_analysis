import pandas as pd
import numpy as np
import sys, getopt
import scipy.interpolate
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

def main(inputfile_surv, inputfile_feature, outputfile_LTV, outputfile_feature, discount_rate = 0.15):

	survival_series, buckets, counts_in_bucket, daily_margin = pickle.load(open(inputfile_surv,'rb'))

	df = pd.read_csv(inputfile_feature)

	df['LTV'] = 0

	df = df.set_index('user_id')

	LTV_bucket_vals = []
	bucket_medians = []

	for ii, bucket in enumerate(buckets):
		num_days = int(survival_series[ii].index[-1])
		survival_interp = interpolate_survival(survival_series[ii], num_days)
		users_temp = list(df[df['use_buckets']==bucket].index)
		for user in users_temp:
			margin = df.ix[user, 'total_order_value']/365.
			# margin = df.ix[user, 'total_order_value']/float(df.ix[user, 'duration'])
			df.ix[user, 'LTV'] = LTV(survival_interp, margin, discount_rate)
		LTV_bucket_vals.append(np.mean(list(df[df['use_buckets']==bucket]['LTV'])))
		bucket_medians.append(np.median(list(df[df['use_buckets']==bucket]['use_count'])))

	pickle.dump((list(df['LTV']), survival_series, LTV_bucket_vals, buckets, bucket_medians, counts_in_bucket, daily_margin), open(outputfile_LTV, 'wb'))

	df.to_csv(outputfile_feature)
	
if __name__ == '__main__':
	main(inputfile_surv, inputfile_feature, outputfile_LTV, outputfile_feature)
