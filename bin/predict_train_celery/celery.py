from __future__ import absolute_import
from celery import Celery

app = Celery(
	'predict_train_celery',
    broker='mongodb://',
    backend='mongodb://',
    include=['predict_train_celery.tasks']
)

# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_RESULT_EXPIRES = 3600,
)

if __name__ == '__main__':
    app.start()
