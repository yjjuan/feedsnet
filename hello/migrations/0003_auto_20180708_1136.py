# Generated by Django 2.0.5 on 2018-07-08 03:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hello', '0002_auto_20180708_1007'),
    ]

    operations = [
        migrations.AlterField(
            model_name='examiner',
            name='model_time',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='examiner',
            name='pend_outcome_time',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='examiner',
            name='pend_time',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='examiner',
            name='test_accu',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='examiner',
            name='train_accu',
            field=models.CharField(max_length=100),
        ),
    ]