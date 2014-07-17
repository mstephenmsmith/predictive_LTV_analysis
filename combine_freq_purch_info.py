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

	inputfile_split = inputfile.split()
	print 'Input frequency use file is "', inputfile_split[0]
	print 'Input purchase file is "', inputfile_split[1]
	print 'Output file is "', outputfile

	df_freq_use = pd.read_csv(inputfile_split[0])
	df_purch = pd.read_csv(inputfile_split[1])

	output_df = pd.merge(df_freq_use, df_purch, on = 'user_id', how = 'left')

	output_df.to_csv(outputfile)

if __name__ == '__main__':
	main(sys.argv[1:])