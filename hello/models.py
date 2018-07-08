from django.db import models

# Create your models here.
class Greeting(models.Model):
    when = models.DateTimeField('date created', auto_now_add=True)

class Examiner(models.Model):
    model_name = models.CharField(max_length = 100)
    model_time = models.DateTimeField('date created', auto_now=True)
    train_accu = models.DecimalField(max_digits=4, decimal_places =2)
    test_accu = models.DecimalField(max_digits=4, decimal_places =2)
    pend_time = models.DateTimeField('pending on', auto_now=True)
    pend_outcome_time = models.DateTimeField('checked on', auto_now=True)