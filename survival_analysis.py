import pandas as pd
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt
import cPickle as pickle
from lifelines.estimation import KaplanMeierFitter

def get_churn_data(df, min_date, max_date, time_to_churn = 1):
	
	ns_in_day = float(8.64*10**13)
	
	max_date_overall = df[max_date].max()
	
	diff_from_max_date = df[max_date].apply(lambda x: max_date_overall - x)

	churn = np.where(diff_from_max_date.apply(lambda x: x.item()/ns_in_day) > time_to_churn,1,0)
	
	last_date_of_interaction = df[max_date]

	last_date_of_interaction.iloc[np.where(diff_from_max_date.apply(lambda x: x.item()/ns_in_day) < time_to_churn)[0]] = max_date_overall
	
	length_of_life = (last_date_of_interaction - df[min_date]).apply(lambda x: x.item()/ns_in_day)
	
	df['duration'] = list(length_of_life)
	df['churn'] = list(churn)
	
	return df


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

	df = pd.read_csv(inputfile)
	df = df.dropna()

	df['first_purch_date'] = pd.to_datetime(df['first_purch_date'])
	df['last_purch_date'] = pd.to_datetime(df['last_purch_date'])

	all_churn_data = get_churn_data(df, "first_purch_date", "last_purch_date")

	all_churn_data_gt0 = all_churn_data[(all_churn_data.duration > 0) & (all_churn_data.std_freq >=0) & (all_churn_data.mean_freq > 0)]

	# buckets = [0,7,14,30,90,400]
	# all_churn_data_gt0["freq_bucket"] = all_churn_data_gt0["mean_freq"]
	# all_churn_data_gt0.freq_bucket = pd.cut(all_churn_data_gt0.freq_bucket, buckets)

	buckets = [0,10,20,30,50,75,100,200,np.max(all_churn_data_gt0["hukk_count"])]
	all_churn_data_gt0["hukk_buckets"] = all_churn_data_gt0["hukk_count"]
	all_churn_data_gt0 = all_churn_data_gt0.sort(columns = ["hukk_buckets"])
	all_churn_data_gt0.hukk_buckets = pd.cut(all_churn_data_gt0.hukk_buckets, buckets)

	xx = pd.cut(all_churn_data_gt0.hukk_buckets, buckets)

	#print "Levels ",all_churn_data_gt0.hukk_buckets.levels

	# unique_buckets = list(set(all_churn_data_gt0.freq_bucket.order()))[1:]

	unique_buckets = list(xx.levels)

	kmf_buckets = []

	avg_durations = []
	avg_total_spent = []
	daily_margin = []
	counts_in_bucket = []

	for bucket in unique_buckets:
		# indices_ = np.where(all_churn_data_gt0.freq_bucket == bucket)
		indices_ = np.where(all_churn_data_gt0.hukk_buckets == bucket)

		counts_in_bucket_temp = all_churn_data_gt0[all_churn_data_gt0.hukk_buckets == bucket].hukk_count.count()

		counts_in_bucket.append(counts_in_bucket_temp)

		avg_durations_temp = all_churn_data_gt0[all_churn_data_gt0.hukk_buckets == bucket].duration.mean()

		avg_durations.append(avg_durations_temp)

		avg_total_spent_temp = all_churn_data_gt0[all_churn_data_gt0.hukk_buckets == bucket].total_order_value.mean()

		avg_total_spent.append(avg_total_spent_temp)

		daily_margin.append(avg_total_spent_temp/float(avg_durations_temp))

		T = all_churn_data_gt0['duration'].iloc[indices_]
		C = all_churn_data_gt0['churn'].iloc[indices_]
		
		kmf = KaplanMeierFitter()
		kmf.fit(T, event_observed = C, label=bucket)
		kmf_buckets.append(kmf)

	# for jj, kmf_ in enumerate(kmf_buckets):
	# 	print kmf_, kmf_.median_
	# 	if jj==0:
	# 		ax = kmf_.plot()
	# 	else:
	# 		kmf_.plot(ax=ax)

	kmf_values = [x.survival_function_ for x in kmf_buckets]

	pickle.dump((kmf_values, unique_buckets, counts_in_bucket, daily_margin), open('kmf_models.p', 'wb')) 

if __name__ == '__main__':
 	main(sys.argv[1:])