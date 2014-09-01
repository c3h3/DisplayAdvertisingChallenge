from imputers.tasks import imputer_train
from imputers.tasks import My_add
from imputers.tasks import My_mul

#results = [imputer_train.apply_async(kwargs={"col_ind":i}) for i in range(39)]
for i in range(1000):
	for j in range(1000):
		a = My_add.apply_async(kwargs={"x":i, "y":j})
		print 'a'
		b = My_mul.apply_async(kwargs = {'x':i, "y":j})
		print "b"