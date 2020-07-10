from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class signup(models.Model):
    id=models.EmailField(unique=True,primary_key=True)
    phoneNumber=models.DecimalField(max_digits=10,decimal_places=0,blank=True)
    name=models.CharField(max_length=100,null=True,blank=True)
    registered_at=models.DateTimeField(auto_now=True)
    dob=models.DateField(null=True,blank=True)
    image=models.FileField(null=True,upload_to='Images/',default='Images/None/NO_img.jpg')
    gender=models.CharField(max_length=1,null=True)
    password=models.CharField(null=False,max_length=100,blank=True)
    verified=models.CharField(default='no',max_length=3,blank=True,null=True)
    otp=models.DecimalField(max_digits=6,decimal_places=0,blank=True,null=True)


