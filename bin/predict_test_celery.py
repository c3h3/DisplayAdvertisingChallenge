from hunkaggle.criteo import tools
from hunkaggle.criteo.settings import SOURCE_DATA_DIR
from hunkaggle.criteo.settings import TRAINING_COLUMN_NAMES
from hunkaggle.criteo.settings import TESTING_COLUMN_NAMES
import blz, os
import numpy as np

from predict_test_celery.tasks import modelPredictor
from settings import *
try:
    import cPickle as pickle
except:
    import pickle

model_path = SOURCE_DATA_DIR + "models.pickle"
with open(model_path, "rb") as rf:
    models = pickle.load(rf)

num_models = len(models)
print num_models

jobs = [(model_path, ind, tools.TESTING_BLZ_PATH, TESTING_COLUMN_NAMES) for ind in range(104, num_models)]

for job in jobs:
    modelPredictor.delay(job) 
