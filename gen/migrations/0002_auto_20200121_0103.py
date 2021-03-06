# Generated by Django 2.2.2 on 2020-01-20 19:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gen', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='signup',
            name='dob',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='signup',
            name='gender',
            field=models.CharField(blank=True, max_length=1, null=True),
        ),
        migrations.AlterField(
            model_name='signup',
            name='name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='signup',
            name='otp',
            field=models.DecimalField(blank=True, decimal_places=0, max_digits=6, null=True),
        ),
        migrations.AlterField(
            model_name='signup',
            name='phoneNumber',
            field=models.DecimalField(blank=True, decimal_places=0, max_digits=10),
        ),
    ]
