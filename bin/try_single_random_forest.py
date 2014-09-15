
if __name__ == "__main__":
    from hunkaggle.criteo.models import *

    total_n = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[0])).shape[0]
    m_groups = 40
    dived_n = total_n / m_groups if total_n % m_groups == 0 else total_n / m_groups + 1
    training_data_slices = map(lambda xx:slice(*xx),tools.get_separation_pairs(total_n,dived_n))
    
    from sklearn.ensemble import RandomForestClassifier
    
    one_slice = training_data_slices[0]
    rf_params = {'min_samples_split': 10, 'n_estimators': 200, "n_jobs":12}
    model_params = {"model_series":"RFmss20ne130-40Groups", 
                    "model_type": RandomForestClassifier, 
                    "model_parameters":rf_params, 
                    "data_slice":one_slice,
                    "predict_limit_instances":5000000}

    create_new_model_with_origin_training_data(**model_params)

