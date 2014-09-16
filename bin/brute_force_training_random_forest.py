from hunkaggle.criteo.models import *

total_n = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[0])).shape[0]
m_groups = 40
dived_n = total_n / m_groups if total_n % m_groups == 0 else total_n / m_groups + 1
training_data_slices = map(lambda xx:slice(*xx),tools.get_separation_pairs(total_n,dived_n))

# from hunkaggle.criteo.settings import MAX_N_DATA_COLUMN_DIVIDERS
# from multiprocessing import Pool 
# MAX_N_DATA_COLUMN_DIVIDERS = 4


import datetime
global_tic = datetime.datetime.now()
print "[GLOBAL LOG] global_tic = ",global_tic

 
def _create_new_model_with_origin_training_data(one_slice):
    
    
    from sklearn.ensemble import RandomForestClassifier
    rf_params = {'min_samples_split': 10, 'n_estimators': 200, "n_jobs":12}
    model_params = {"model_series":"RFmss%sne%s-%sGroups" % (rf_params["min_samples_split"], rf_params["n_estimators"], m_groups), 
                    "model_type": RandomForestClassifier, 
                    "model_parameters":rf_params, 
                    "data_slice":one_slice,
                    "predict_limit_instances":5000000,
                    "prediction_methond": "predict_proba"}
    create_new_model_with_origin_training_data(**model_params)
 

    

for one_slice in training_data_slices:
    
    print "[GLOBAL LOG] training slice:", one_slice
    one_slice_tic = datetime.datetime.now()
    print "[GLOBAL LOG] one_slice_tic = ", one_slice_tic
    
    _create_new_model_with_origin_training_data(one_slice)
    
    print "[GLOBAL LOG] one_slice cost time / sec = ", (datetime.datetime.now() - one_slice_tic).seconds
    

print "[GLOBAL LOG] global cost time / sec = ", (datetime.datetime.now() - global_tic).seconds


# p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
# p.map(_create_new_model_with_origin_training_data, training_data_slices)
#   
# p.close()
# p.join()
