import pandas as pd
import numpy as np
import sys, getopt

def main(inputfile_freq, inputfile_purch, outputfile):

	df_freq_use = pd.read_csv(inputfile_freq)
	df_purch = pd.read_csv(inputfile_purch)

	output_df = pd.merge(df_freq_use, df_purch, on = 'user_id', how = 'left')

	output_df.to_csv(outputfile)

if __name__ == '__main__':
	main(inputfile_freq, inputfile_purch, outputfile)