
from hunkaggle.criteo import tools
from hunkaggle.criteo.settings import TRAINING_COLUMN_NAMES
import blz, os
import numpy as np
from hunkaggle.criteo.settings import MODELS_PATH


import uuid
try:
    import cPickle as pickle
except:
    import pickle
    

DEFAULT_LIMIT_INSTANCE = 5000000
PRINT_MESSAGE_FORMAT = "[{model_id}] {message}"



class Model(object):
    # Change self.model_home to self.MODELS_HOME
    MODELS_HOME = MODELS_PATH 
    
    
    def __init__(self, model_series = "TryModels", model_id=None, model_home = MODELS_PATH):
        
        self.model_series = model_series
        self.model_home = model_home
        
        if model_id==None:
            self.create_new_model_id()
        else:
            self.model_id = self.model_id = self.model_series + "_" + model_id
            self.model_path = os.path.join(self.model_home, self.model_id)
        
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        
        self.all_training_prediction_path = os.path.join(self.model_path, "all_training_prediction")
        if not os.path.exists(self.all_training_prediction_path):
            os.mkdir(self.all_training_prediction_path)
        
        self.all_testing_prediction_path = os.path.join(self.model_path, "all_testing_prediction")
        if not os.path.exists(self.all_testing_prediction_path):
            os.mkdir(self.all_testing_prediction_path)
        
        
    @property
    def all_models_in_series(self):
        return [one_model for one_model in os.listdir(self.model_home) if one_model.startswith(self.model_series)]
    
    
    def create_new_model_id(self):
        
        self.model_id = self.model_series + "_" + str(uuid.uuid1())
        self.model_path = os.path.join(self.model_home, self.model_id)
        
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
        return self.model_id
    
    
    def set_training_model_type(self, model_type):
        self.model_type = model_type
        
        
    def set_training_model_parameters(self, **parameters):
        assert "model_type" in self.__dict__
        self.model_parameters = parameters
        
    
    def set_training_data_slice(self, data_slice):
        self.training_data_slices = data_slice
        
        
    def set_training_model(self, model):
        self.model = model
        self.model_type = type(model)
        self.all_model_parameters = model.get_params()
        
    
    def fit_model(self, append_self=True, feature_columns=TRAINING_COLUMN_NAMES[2:], label_columns=TRAINING_COLUMN_NAMES[1]):
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] get features ... ")
        
        self.feature_columns = [one_col for one_col in TRAINING_COLUMN_NAMES if one_col in feature_columns]
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] get labels ... ")
        
        if isinstance(label_columns,(str,unicode)):
            self.label_columns = [label_columns]
        
        read_columns = self.label_columns + self.feature_columns
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] build features blz ... ")
        
        training_cols_blz = map(lambda xx: blz.open(os.path.join(tools.TRAINING_BLZ_PATH,xx)),read_columns)
        
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] get training_data_slices from blz ... ")
        
        if "training_data_slices" in self.__dict__:
            sample_data_arr = np.c_[map(lambda xx:xx[self.training_data_slices],training_cols_blz)].T
        else:
            sample_data_arr = np.c_[map(lambda xx:xx[0:],training_cols_blz)].T
            
        X = sample_data_arr[:,1:]
        y = sample_data_arr[:,0]
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] init model ... ")
        
        if "all_model_parameters" in self.__dict__:
            model = self.model_type(**self.all_model_parameters)
        
        elif "model_parameters" in self.__dict__:
            model = self.model_type(**self.model_parameters)
        
        else:
            model = self.model_type()
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] save model_info ... ")
        
        self.save_model_info()
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[fit_model] fit model and return ... ")
        
        if append_self:
            self.model = model.fit(X,y)
            return self.model
        else:
            return model.fit(X,y)
    
    
#     def predict_all_training_data(self, prediction_methond="decision_function",limit_instances=1000000):
#         pass
    
        
    def predict_data(self, prediction_methond="decision_function", limit_instances=1000000, on_which_data="training"):
    
        assert "feature_columns" in self.__dict__
        assert "model" in self.__dict__
        assert prediction_methond in dir(self.model)
        assert on_which_data in ["training","testing"]
        
        read_columns = self.feature_columns
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] read predict_features_blz ... ")
        
        if on_which_data == "training":
            prediction_results_blz_path = os.path.join(self.all_training_prediction_path, prediction_methond)
            predict_features_blz_path_list = map(lambda xx: os.path.join(tools.TRAINING_BLZ_PATH,xx),read_columns)
        else:
            prediction_results_blz_path = os.path.join(self.all_testing_prediction_path, prediction_methond)
            predict_features_blz_path_list = map(lambda xx: os.path.join(tools.TESTING_BLZ_PATH,xx),read_columns)
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] checking prediction results ... ")
            
        if os.path.exists(prediction_results_blz_path):
            print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] there already exist prediction results ... ")
            return blz.open(prediction_results_blz_path)
        else:
            
            print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] there is no prediction result ... ")
            print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] load predict_features_blz ... ")
            
            predict_features_blz = map(blz.open, predict_features_blz_path_list)
            
            print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] load prediction_function ... ")
            
            prediction_function = getattr(self.model,prediction_methond)
        
        
            #predict_features_blz = map(lambda xx: blz.open(os.path.join(tools.TRAINING_BLZ_PATH,xx)),read_columns)
        
            totalN = predict_features_blz[0].shape[0]
            divideN = limit_instances
        
            all_pairs = tools.get_separation_pairs(totalN, divideN)
        
            one_pair = all_pairs[0]
            print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] predict data slice %s ~ %s " % tuple(one_pair))
        
        
            predict_features_arr = np.c_[map(lambda xx:xx[one_pair[0]:one_pair[1]],predict_features_blz)].T 
        
            prediction_results_blz = blz.barray(prediction_function(predict_features_arr),
                                            rootdir=prediction_results_blz_path)
        
            for one_pair in all_pairs[1:]:
                print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] predict data slice %s ~ %s " % tuple(one_pair))
        
                predict_features_arr = np.c_[map(lambda xx:xx[one_pair[0]:one_pair[1]],predict_features_blz)].T  
            
                prediction_results_blz.append(prediction_function(predict_features_arr))
                del predict_features_arr
                
            prediction_results_blz.flush()
            print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[predict_data] finished prediction_results_blz.flush() and return")
    
        
            return prediction_results_blz
    
    
    def list_all_predictions(self):
        return {"training":os.listdir(self.all_training_prediction_path),
                "testing":os.listdir(self.all_testing_prediction_path)}
    
    
    def load_prediction_blz(self, datatype="training", valuetype="decision_function", limit_instances=1000000):
        assert datatype in ["training","testing"]
        if datatype == "training":
            loading_blz_path = os.path.join(self.all_training_prediction_path,valuetype)
        else:
            loading_blz_path = os.path.join(self.all_testing_prediction_path,valuetype)
        
        #eturn os.path.exists(loading_blz_path), loading_blz_path
    
        if os.path.exists(loading_blz_path):
            return blz.open(loading_blz_path)
        
            
    def save_model_info(self):
        model_pickle_file_path = os.path.join(self.model_path,"model_info.pickle")
        model_info = {}
        
        model_info["model_type"] = self.model_type
        model_info["training_data_slices"] = self.training_data_slices
        
        if "training_dataset" in self.__dict__:
            model_info["training_dataset"] = self.training_dataset
        else:
            model_info["training_dataset"] = "origin"

        
        
        if "model" in self.__dict__:
            model_info["model_paramters"] = self.model.get_params()
            
        else:
            if "all_model_parameters" in self.__dict__:
                model_info["model_paramters"] = self.all_model_parameters
            else:
                model_info["model_paramters"] = self.model_parameters
                
            
        if "feature_columns" in self.__dict__:
            model_info["feature_columns"] = self.feature_columns
        else:
            model_info["feature_columns"] = TRAINING_COLUMN_NAMES[2:]
        
        
        if "additional_feature_columns" in self.__dict__:
            model_info["additional_feature_columns"] = self.additional_feature_columns
        else:
            model_info["additional_feature_columns"] = []
        
            
        
        with open(model_pickle_file_path, "wb") as wf:
            pickle.dump(model_info,wf)
            
        return model_info
        
        
        
def create_new_model_with_origin_training_data(model_series, model_type, model_parameters={}, 
                                               feature_columns=TRAINING_COLUMN_NAMES[2:], 
                                               data_slice=slice(-1000000,None,None), 
                                               prediction_methond = "decision_function",
                                               predict_limit_instances = 1000000,
                                               predict_on = ["training","testing"]):
    
    model = Model(model_series=model_series)
    
    model.set_training_model_type(model_type)
    model.set_training_model_parameters(**model_parameters)
    model.set_training_data_slice(data_slice)
    
    model.fit_model(feature_columns=feature_columns)
    
    for one_dataset in predict_on:
        model.predict_data(prediction_methond=prediction_methond, 
                           limit_instances=predict_limit_instances, 
                           on_which_data=one_dataset)
    
    
    
    
    
    
    