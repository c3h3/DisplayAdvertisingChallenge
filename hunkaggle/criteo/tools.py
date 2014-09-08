


from settings import TRAINING_DATA_PATH, TESTING_DATA_PATH
from settings import TRAINING_BLZ_PATH, TESTING_BLZ_PATH

import numpy as np
import blz, os


def get_one_column_data_list(colname, data_csv_path=TRAINING_DATA_PATH):
    
    with open(data_csv_path,"r") as rf:
        colnames_line = rf.readline()
        colnames = colnames_line.strip().split(",")
        colnames_array = np.array(colnames)
        data_col_idx = np.where(colnames_array == colname)[0][0]
        
        data_list = []
        one_line_str = "temp_str"

        while one_line_str != "":
            one_line_str = rf.readline()
            one_line_str_data = one_line_str.strip().split(",")
            if len(one_line_str_data) == len(colnames):
                data_list.append(one_line_str.strip().split(",")[data_col_idx])
            
    return data_list


def convert_column_to_blz(colname):
    
    if not colname in os.listdir(TRAINING_BLZ_PATH):
    
        if colname.startswith("C"):
            
            train_data = get_one_column_data_list(colname, data_csv_path=TRAINING_DATA_PATH)
            test_data = get_one_column_data_list(colname, data_csv_path=TESTING_DATA_PATH)
            
            train_data_has_missing_values = "" in train_data
            test_data_has_missing_values = "" in test_data
            
            
            all_data_np_array = np.array(train_data + test_data)
            
            uu, ui, ii = np.unique(all_data_np_array,return_index=True,return_inverse=True)
            
            train_blz_root = os.path.join(TRAINING_BLZ_PATH,colname)
            test_blz_root = os.path.join(TESTING_BLZ_PATH,colname)
            
            train_barr = blz.barray(ii[:len(train_data)],rootdir=train_blz_root)
            train_barr.attrs["states"] = uu.tolist()
            train_barr.attrs["has_missings"] = train_data_has_missing_values
            
            test_barr = blz.barray(ii[len(train_data):],rootdir=test_blz_root)
            test_barr.attrs["states"] = uu.tolist()
            test_barr.attrs["has_missings"] = test_data_has_missing_values
            
        else:
            
            if colname in ["Id","Label"]:
                convert_type = int
            else:
                convert_type = np.float64
            
            default_missing_value = 1.0e10
                
            
            
            data = get_one_column_data_list(colname, data_csv_path=TRAINING_DATA_PATH)
            has_missing = "" in data
            missing_value = convert_type(default_missing_value)
            if has_missing:
                int_data = map(lambda xx: missing_value if xx in [""] else convert_type(xx), data)
            else:
                int_data = map(convert_type, data)
                
            blz_root = os.path.join(TRAINING_BLZ_PATH,colname)
            barr = blz.barray(int_data,rootdir=blz_root)
            barr.attrs["missing_value"] = int(missing_value)
            barr.attrs["has_missing"] = has_missing    
            
            if colname != "Label":
                data = get_one_column_data_list(colname, data_csv_path=TESTING_DATA_PATH)
                has_missing = "" in data
                missing_value = convert_type(default_missing_value)
                if has_missing:
                    int_data = map(lambda xx: missing_value if xx in [""] else convert_type(xx), data)
                else:
                    int_data = map(convert_type, data)
                
                blz_root = os.path.join(TESTING_BLZ_PATH,colname)
                barr = blz.barray(int_data,rootdir=blz_root)
                barr.attrs["missing_value"] = int(missing_value)
                barr.attrs["has_missing"] = has_missing
        
