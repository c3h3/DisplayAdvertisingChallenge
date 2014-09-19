
from hunkaggle.criteo.models import *

total_n = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[0])).shape[0]
m_groups = 1
dived_n = total_n / m_groups if total_n % m_groups == 0 else total_n / m_groups + 1
training_data_slices = map(lambda xx:slice(*xx),tools.get_separation_pairs(total_n,dived_n))

# from hunkaggle.criteo.settings import MAX_N_DATA_COLUMN_DIVIDERS
# from multiprocessing import Pool 
 

import datetime
global_tic = datetime.datetime.now()
print "[GLOBAL LOG] global_tic = ",global_tic

 
def _create_new_model_with_origin_training_data(one_slice):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import SGDClassifier
    base_estimator = SGDClassifier(loss="log")
    model_params = {"model_series":"BaggingClassifier", 
                    "model_type": BaggingClassifier, 
                    "model_parameters":{"n_estimators":20,"n_jobs":12}, 
                    "data_slice":one_slice,
                    "predict_limit_instances":5000000}
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
# p.close()
# p.join()
