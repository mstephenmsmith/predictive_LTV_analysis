import time
import numpy as np

import frequencies
import purchase_info
import combine_freq_purch_info
import survival_analysis
import lifetime_value
import plotting
import model_pred

start_time = time.time()

folder_path = '/Users/mstephenmsmith/Zipfian/capstone_project/data/'

freq_file_input = 'hukkster_hukks_query_071414.csv'
freq_file_output = 'hukkster_frequencies_v2.csv'
purch_info_input = 'all_hukksters.csv'
purch_info_output = 'purch_info_v2.csv'
combine_freq_purch_info_output = 'combined_hukk_table.csv'
survival_analysis_output = 'kmf_models.p'
lifetime_value_output = 'ltv_survival_models.p'
surv_feature_matrix = 'surv_feature_matrix.csv'
final_feature_matrix = 'final_feature_matrix.csv'

user_attributes = 'user_attributes.csv'

model_save = "model_save.p"




buckets = [0,10,20,30,50,75,100,200,100000]
#buckets = [0,100000]
time_to_churn = 10


#frequencies.main(folder_path+freq_file_input, folder_path+freq_file_output)
# purchase_info.main(folder_path+purch_info_input, folder_path+purch_info_output)
# combine_freq_purch_info.main(folder_path+freq_file_output, folder_path+purch_info_output, folder_path+combine_freq_purch_info_output)


# survival_analysis.main(folder_path+combine_freq_purch_info_output, folder_path+survival_analysis_output,buckets,time_to_churn)
# lifetime_value.main(folder_path+survival_analysis_output, folder_path+surv_feature_matrix, folder_path+lifetime_value_output, folder_path+final_feature_matrix)

model_pred.main(folder_path+final_feature_matrix, folder_path+user_attributes,folder_path+model_save)


plotting.main(folder_path+lifetime_value_output,folder_path+model_save)



# print time.time() - start_time