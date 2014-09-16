from hunkaggle.criteo.models import *

total_n = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[0])).shape[0]
m_groups = 40
dived_n = total_n / m_groups if total_n % m_groups == 0 else total_n / m_groups + 1
training_data_slices = map(lambda xx:slice(*xx),tools.get_separation_pairs(total_n,dived_n))

# from hunkaggle.criteo.settings import MAX_N_DATA_COLUMN_DIVIDERS
from multiprocessing import Pool 
 
MAX_N_DATA_COLUMN_DIVIDERS = 4
 
def _create_new_model_with_origin_training_data(one_slice):
    from sklearn.ensemble import RandomForestClassifier
    rf_params = {'min_samples_split': 20, 'n_estimators': 130, "n_jobs":3}
    model_params = {"model_series":"RFmss20ne130-40Groups", 
                    "model_type": RandomForestClassifier, 
                    "model_parameters":rf_params, 
                    "data_slice":one_slice,
                    "predict_limit_instances":1000000}
    create_new_model_with_origin_training_data(**model_params)
 
p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
p.map(_create_new_model_with_origin_training_data, training_data_slices)
  
p.close()
p.join()