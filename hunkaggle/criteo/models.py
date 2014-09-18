
from hunkaggle.criteo import tools
from hunkaggle.criteo.settings import TRAINING_COLUMN_NAMES, TESTING_COLUMN_NAMES
import blz, os
import numpy as np
from hunkaggle.criteo.settings import MODELS_PATH, SUBMITS_PATH, AVG_MODELS_PATH
from scipy.special import erf

import uuid
import datetime

try:
    import cPickle as pickle
except:
    import pickle
    
import scipy as sp


DEFAULT_LIMIT_INSTANCE = 5000000
PRINT_MESSAGE_FORMAT = "[{model_id}] {message}"



def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sp.sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def llfun_npsum(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = np.sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def blz_llvec_fun(act, pred, output_vector_type ="numpy"):
    epsilon = 1e-15
    pred[pred<epsilon] = epsilon
    pred[pred > (1-epsilon) ] = 1-epsilon
    bt = blz.btable([act,pred],names=["y","py"])
    if output_vector_type =="numpy":
        return bt.eval("-y*log(py) - (1-y)*log(1-py)",out_flavor="numpy")
    else:
        return bt.eval("-y*log(py) - (1-y)*log(1-py)")
    
def ll_vec(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    return -act*sp.log(pred) - sp.subtract(1,act)*sp.log(sp.subtract(1,pred))




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
        
    
    def __unicode__(self):
        if self.has_model_info:
            return self.model_id + " : " + unicode(self.model_info)
        else:
            return self.model_id
    
    
    def __repr__(self):
        return self.__unicode__()
        
    
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
    
    @property
    def list_all_predictions(self):
        return {"training":os.listdir(self.all_training_prediction_path),
                "testing":os.listdir(self.all_testing_prediction_path)}
    
    
    def load_prediction_blz(self, datatype="training", valuetype="decision_function", limit_instances=1000000):
        assert datatype in ["training","testing"]
        if datatype == "training":
            loading_blz_path = os.path.join(self.all_training_prediction_path,valuetype)
        else:
            loading_blz_path = os.path.join(self.all_testing_prediction_path,valuetype)
        
        #return os.path.exists(loading_blz_path), loading_blz_path
    
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
    
    
    @property
    def has_model_info(self):
        return "model_info.pickle" in os.listdir(self.model_path)
    
    
    @property
    def model_info(self):
        assert "model_info.pickle" in os.listdir(self.model_path)
        model_pickle_file_path = os.path.join(self.model_path,"model_info.pickle")
        with open(model_pickle_file_path, "rb") as wf:
            model_info = pickle.load(wf)
            
        return model_info
    

    
    def compute_training_data_logloss(self, return_type="value", 
                                      use_prediction="predict_proba", 
                                      prob_func=lambda xx:erf(np.subtract(xx,-0.9)),
                                      output_vector_type = "numpy",
                                      sample_slice  = slice(0,None,None)
                                      ):
    
        assert return_type in ["value", "vector"]
        assert output_vector_type in ["blz","numpy"]
        assert use_prediction in self.list_all_predictions["training"]
        
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[compute_training_data_logloss] conpute logloss with %s " % use_prediction)
        
        use_prediction_blz = self.load_prediction_blz(datatype="training", valuetype=use_prediction)
        
        
        if use_prediction == "predict_proba":
            prediction_prob = use_prediction_blz[sample_slice,1]
        else:
            #_vec_prob_func = np.vectorize(prob_func)
            #prediction_prob = np.apply_along_axis(prob_func,0,use_prediction_blz[0:])
            prediction_prob = prob_func(use_prediction_blz[sample_slice])
            
            #prediction_prob = blz.eval("prob_func(use_prediction_blz)", vm="python")
        
        #return prediction_prob
        
        exact_ans = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[1]))[sample_slice]
        
#         exact_ans = blz.open(os.path.join(tools.TRAINING_BLZ_PATH,TRAINING_COLUMN_NAMES[1]))
#         bt = blz.btable(columns=[prediction_prob,exact_ans],names=["y","py"])
        
        logloss_vector = blz_llvec_fun(exact_ans,prediction_prob,output_vector_type)
        logloss_value = logloss_vector.sum()/logloss_vector.shape[0]
        print PRINT_MESSAGE_FORMAT.format(model_id=self.model_id,message="[compute_training_data_logloss] logloss_value = %s " % logloss_value)
        
        if return_type == "value":
            return logloss_value
        else:
            return logloss_vector
    
    
    def create_kaggle_submit_csv(self, submit_format="%d,%.6f"):
        assert "predict_proba" in self.list_all_predictions["testing"]
        prediction_prob = self.load_prediction_blz(datatype="testing", valuetype="predict_proba")[:,1]
        ids_barray = blz.open(os.path.join(tools.TESTING_BLZ_PATH,TESTING_COLUMN_NAMES[0]))
        bt = blz.btable(columns=[ids_barray,prediction_prob], names=["Id","Predicted"])
        all_results = [submit_format % tuple(xx) for xx in bt.iter()]
        all_results_string = "\n".join([",".join(bt.names)] + all_results)
        
        submit_filename = "%s_%s.csv" % (self.model_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        
        submit_filepath = os.path.join(SUBMITS_PATH,submit_filename)
        
        with open(submit_filepath,"w") as wf:
            wf.write(all_results_string)
        
        
            
class ModelSeries(object):
    def __init__(self, series_name, series_home):
        self.series_name = series_name
        self.series_home = series_home
    
    
    @property
    def series_model_ids(self):
        return map(lambda xx:xx.replace(self.series_name+"_",""),os.listdir(self.series_home))
        
        
    @property
    def series_models(self):
        return [Model(model_series=self.series_name, 
                      model_id=one_id, 
                      model_home = self.series_home) for one_id in self.series_model_ids]
        
    
    def compute_training_data_logloss(self,**kwargs):
        logloss_dict = {}
        
        for one_model in self.series_models:
            logloss_dict[one_model.model_id] = one_model.compute_training_data_logloss(**kwargs)
        return logloss_dict


        
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
    
    
    
    
    
    
    