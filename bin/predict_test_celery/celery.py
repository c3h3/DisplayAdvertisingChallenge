from __future__ import absolute_import
from celery import Celery

app = Celery(
	'predict_test_celery',
    broker='mongodb://220.133.188.246',
    backend='mongodb://220.133.188.246',
    include=['predict_test_celery.tasks']
)

# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_RESULT_EXPIRES = 3600,
)

if __name__ == '__main__':
    app.start()
