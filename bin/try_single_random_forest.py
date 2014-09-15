
if __name__ == "__main__":
#     from hunkaggle.criteo.models import *
# 
#     total_n = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[0])).shape[0]
#     m_groups = 40
#     dived_n = total_n / m_groups if total_n % m_groups == 0 else total_n / m_groups + 1
#     training_data_slices = map(lambda xx:slice(*xx),tools.get_separation_pairs(total_n,dived_n))
#     
#     from sklearn.ensemble import RandomForestClassifier
#     
#     one_slice = training_data_slices[0]
#     rf_params = {'min_samples_split': 10, 'n_estimators': 200, "n_jobs":12}
#     model_params = {"model_series":"RFmss20ne130-40Groups", 
#                     "model_type": RandomForestClassifier, 
#                     "model_parameters":rf_params, 
#                     "data_slice":one_slice,
#                     "predict_limit_instances":5000000}
# 
#     create_new_model_with_origin_training_data(**model_params)
    
    import datetime
    tic = datetime.datetime.now()
    print "tic = ",tic
    
    from hunkaggle.criteo import tools
    from hunkaggle.criteo.settings import TRAINING_COLUMN_NAMES
    import blz, os
    import numpy as np
    
    print "finished import ... ", (datetime.datetime.now() - tic).seconds
    print "loading data ... "
    
    training_cols = map(lambda xx: blz.open(os.path.join(tools.TRAINING_BLZ_PATH,xx)),TRAINING_COLUMN_NAMES[1:])
    sample_data = map(lambda xx:xx[-1000000:],training_cols)
    sample_data_arr = np.c_[sample_data].T
    X = sample_data_arr[:,1:]
    y = sample_data_arr[:,0]
    
    print "finished loading data ... ", (datetime.datetime.now() - tic).seconds
    print "training model "
    
    def training_model(X,y):
        para2 = {'min_samples_split': 20, 'n_estimators': 170}
        from sklearn.ensemble import RandomForestClassifier
    
    
        rfc = RandomForestClassifier(n_jobs=12, **para2)
        rfc.fit(X,y)
    
    training_model(X,y)
    
    print "finished training model ... ", (datetime.datetime.now() - tic).seconds
    
    