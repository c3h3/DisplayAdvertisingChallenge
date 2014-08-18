
import commands

from column_analysis_tooles import TAR_TRAINING_COLUMNS_PATH, TRAINING_COLUMNS_PATH, list_all_data_columns

from settings import MAX_N_DATA_COLUMN_DIVIDERS


from multiprocessing import Pool   

import os

def tar_one_data_column_pickle(colname):
    os.chdir(TRAINING_COLUMNS_PATH)
    print commands.getoutput("tar cfzv %s.tar.gz %s.col" % (colname, colname)) 
    

p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
p.map(tar_one_data_column_pickle, list_all_data_columns() )

p.close()
p.join()


os.chdir(TRAINING_COLUMNS_PATH)
print commands.getoutput("mv *.tar.gz %s" % TAR_TRAINING_COLUMNS_PATH) 
    
