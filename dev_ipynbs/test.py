from imputers.tasks import imputer_train

results = [imputer_train.apply_async(kwargs={"col_ind":i}) for i in range(39)]