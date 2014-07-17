import pandas as pd
import numpy as np
import sys, getopt

def main(argv):

	inputfile = ''
	outputfile = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print 'test.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	print 'Input file is "', inputfile
	print 'Output file is "', outputfile

	df = pd.read_csv(inputfile)

	df = df.dropna()

	keep = np.where((df.transaction_type == 'purchase') & (df.num_items >= 0) & (df.total_order_value > 0))[0]

	df = df.iloc[keep]

	df_purchase_sum = df[['user_id','num_items','total_order_value','commission_value']].groupby('user_id').sum().reset_index()[['user_id','num_items','total_order_value','commission_value']]
	df_purchase_sum.columns = ['user_id','num_items_purch','total_order_value','commission_value']

	df_purchase_sum.to_csv(outputfile)

if __name__ == '__main__':
	main(sys.argv[1:])