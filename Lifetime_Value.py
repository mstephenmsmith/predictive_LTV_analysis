import pandas as pd
import numpy as np
import sys, getopt
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

	return_LTV


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

	print 'Output file is "', inputfile

	kmf_files, buckets = pickle.load(open(inputfile,'rb'))

	survival_series = [x.survival_function_ for x in kmf_files]

if __name__ == '__main__':
	main(sys.argv[1:])