import pandas as pd
import numpy as np
import sys, getopt
from scipy import stats

def main(inputfile, outputfile):

	df = pd.read_csv(inputfile)

	df = df.dropna()

	keep = np.where((df.transaction_type == 'purchase') & (df.num_items >= 0) & (df.total_order_value > 0))[0]

	df = df.iloc[keep]

	users = list(set(df['user_id']))

	df['transaction_date'] = pd.to_datetime(df['transaction_date'])

	first_purchase_amounts = []

	for user in users:
		tmin = min(df[df['user_id']==user]['transaction_date'])
		first_purchase_amounts.append(df[(df['user_id']==user) & (df['transaction_date']==tmin)]['total_order_value'].iloc[0])

	temp_list = map(list, zip(*[users,first_purchase_amounts]))

	first_purchase_amount_df = pd.DataFrame(temp_list,columns=['user_id','first_purchase_amount'])

	most_used_store = df[['user_id', 'store_id']].groupby('user_id').agg(lambda x: stats.mode(x['store_id'])[0]).reset_index()[['user_id', 'store_id']]
	most_used_store.columns = ['user_id', 'most_used_store']

	df_purchase_sum = df[['user_id','num_items','total_order_value','commission_value']].groupby('user_id').sum().reset_index()[['user_id','num_items','total_order_value','commission_value']]
	df_purchase_sum.columns = ['user_id','num_items_purch','total_order_value','commission_value']

	final_df = pd.merge(df_purchase_sum, first_purchase_amount_df, on='user_id')
	final_df = pd.merge(final_df, most_used_store, on='user_id')

	final_df.to_csv(outputfile)

if __name__ == '__main__':
	main(inputfile, outputfile)