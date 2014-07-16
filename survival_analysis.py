import pandas as pd
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt
from lifelines.estimation import KaplanMeierFitter as kmf

def get_churn_data(df, user_col, date_col, min_date, max_date, time_to_churn = 60):
	
	ns_in_day = float(8.64*10**13)
	
	max_date_overall = df[date_col].max()
	
	diff_from_max_date = max_date[date_col].apply(lambda x: max_date_overall - x)

	churn = np.where(diff_from_max_date.apply(lambda x: x.item()/ns_in_day) > time_to_churn,1,0)
	
	max_date[date_col].iloc[np.where(diff_from_max_date.apply(lambda x: x.item()/ns_in_day) < time_to_churn)[0]] = max_date_overall
	
	length_of_life = (max_date[date_col] - min_date[date_col]).apply(lambda x: x.item()/ns_in_day)
	
	df = pd.DataFrame(list(min_date[date_col]), columns=['min_date'])
	df['max_date'] = list(max_date[date_col])
	df['duration'] = list(length_of_life)
	df['churn'] = list(churn)
	
	return df

# Calculate the mean, median, and standard deviation of the difference
# in sequential time periods for each user

def get_means_medians(df, user_id):

	temp_indices = np.where(df['user_id'] == user_id)
	temp_indices = list(temp_indices[0])

	temp_dates = list(df['created_on'].iloc[temp_indices])

	temp_dates = sorted(temp_dates)

	# temp_dates = [l]

	diffs = []

	for j in xrange(len(temp_dates)-1):
		temp_diff = temp_dates[j+1]-temp_dates[j]
		diffs.append(temp_diff)
	
	diffs = [x.days for x in diffs]

	if np.mean(diffs) != 0:
		std_temp = float(np.std(diffs))/float(np.mean(diffs))
	else:
		std_temp = -1
	
	mean_temp = np.mean(diffs)
	median_temp = np.median(diffs)

	return mean_temp, median_temp, std_temp

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

	df = pd.read_csv(inputfile)

	df['created_on'] = pd.to_datetime(df['created_on']);

	print "Getting max and min creation dates..."

	max_created_on = df[['user_id','created_on']].groupby('user_id').max()

	min_created_on = df[['user_id','created_on']].groupby('user_id').min()

	ns_in_day = float(8.64*10**13)

	all_stds = []
	all_means = []
	all_medians = []

	unique_user_ids = list(set(df['user_id']))

	print len(unique_user_ids)

	print "Getting medians and means..."

	for i, user_id in enumerate(unique_user_ids):
		if i%1000 ==0:
			print i
		mean_temp, median_temp, std_temp = get_means_medians(df, user_id)

		all_stds.append(std_temp)
		all_means.append(mean_temp)
		all_medians.append(median_temp)

	print "Getting churn data...."

	all_churn_data = get_churn_data(df,'user_id','created_on',min_created_on, max_created_on)

	all_churn_data['means'] = all_means
	all_churn_data['medians'] = all_medians
	all_churn_data['stds'] = all_stds

	all_churn_data_gt0 = all_churn_data[all_churn_data.duration > 0]

	T = all_churn_data_gt0['duration']
	C = all_churn_data_gt0['churn']

	kmf.fit(T, event_observed = C)

	print kmf.median_

if __name__ == '__main__':
	main(sys.argv[1:])