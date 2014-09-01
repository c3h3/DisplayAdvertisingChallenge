
from multiprocessing import Pool   

dividing_jobs = [xx for xx in COLUMN_NAMES if xx not in list_all_data_columns()]

print "dividing_jobs = ", dividing_jobs

p = Pool(processes = MAX_N_DATA_COLUMN_DIVIDERS)
p.map(pickle_one_data_column, dividing_jobs)

p.close()
p.join()

