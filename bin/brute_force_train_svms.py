
from hunkaggle.criteo.models import *

total_n = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[0])).shape[0]
m_groups = 40
dived_n = total_n / m_groups if total_n % m_groups == 0 else total_n / m_groups + 1
training_data_slices = map(lambda xx:slice(*xx),tools.get_separation_pairs(total_n,dived_n))

from sklearn.svm import LinearSVC

create_models_parameters = []

for one_slice in training_data_slices:
    create_models_parameters.append({"model_series":"LSVM40Groups", 
                                     "model_type": LinearSVC, 
                                     "model_parameters":{"dual":False}, 
                                     "data_slice":one_slice,
                                     "predict_limit_instances":5000000})


print create_models_parameters

from hunkaggle.criteo.settings import MAX_N_DATA_COLUMN_DIVIDERS
from multiprocessing import Pool 
 
def _create_new_model_with_origin_training_data(para_dict):
    create_new_model_with_origin_training_data(**para_dict)
 
p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
p.map(lambda xx:_create_new_model_with_origin_training_data(**xx), create_models_parameters)
  
p.close()
p.join()
