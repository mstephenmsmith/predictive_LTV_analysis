import pandas as pd
import numpy as np
import sys, getopt
import cPickle as pickle
from lifelines.estimation import KaplanMeierFitter

def get_churn_data(df, min_date, max_date, time_to_churn):
	
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

def kmf_calculation(df, bucket):

	indices_ = np.where(df.use_buckets == bucket)

	T = df['duration'].iloc[indices_]
	C = df['churn'].iloc[indices_]
	
	kmf = KaplanMeierFitter()
	kmf.fit(T, event_observed = C, label=bucket)
	
	return kmf

def main(inputfile, outputfile, buckets, time_to_churn):

	df = pd.read_csv(inputfile)

	df = df.dropna()

	df['first_use_date'] = pd.to_datetime(df['first_use_date'])
	df['last_use_date'] = pd.to_datetime(df['last_use_date'])

	all_churn_data = get_churn_data(df, "first_use_date", "last_use_date", time_to_churn)

	all_churn_data_gt0 = all_churn_data[(all_churn_data.duration > 0) & (all_churn_data.std_freq >=0) & (all_churn_data.mean_freq > 0)]


	all_churn_data_gt0["use_buckets"] = all_churn_data_gt0["use_count"]
	all_churn_data_gt0 = all_churn_data_gt0.sort(columns = ["use_buckets"])
	all_churn_data_gt0.use_buckets = pd.cut(all_churn_data_gt0.use_buckets, buckets)

	xx = pd.cut(all_churn_data_gt0.use_buckets, buckets)

	unique_buckets = list(xx.levels)

	kmf_buckets = []

	avg_durations = []
	avg_total_spent = []
	daily_margin = []
	counts_in_bucket = []

	for bucket in unique_buckets:
		
		indices_ = np.where(all_churn_data_gt0.use_buckets == bucket)

		counts_in_bucket_temp = all_churn_data_gt0[all_churn_data_gt0.use_buckets == bucket].use_count.count()

		counts_in_bucket.append(counts_in_bucket_temp)

		avg_durations_temp = all_churn_data_gt0[all_churn_data_gt0.use_buckets == bucket].duration.mean()

		avg_durations.append(avg_durations_temp)

		avg_total_spent_temp = all_churn_data_gt0[all_churn_data_gt0.use_buckets == bucket].total_order_value.mean()

		avg_total_spent.append(avg_total_spent_temp)

		daily_margin.append(avg_total_spent_temp/float(avg_durations_temp))

		kmf = kmf_calculation(all_churn_data_gt0, bucket)
		
		kmf_buckets.append(kmf)

	kmf_values = [x.survival_function_ for x in kmf_buckets]

	pickle.dump((kmf_values, unique_buckets, counts_in_bucket, daily_margin), open(outputfile, 'wb'))

	df_final = all_churn_data_gt0

	df_final.to_csv('./data/surv_feature_matrix.csv')

if __name__ == '__main__':
 	main(inputfile, outputfile, buckets, time_to_churn)
