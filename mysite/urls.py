"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import *
urlpatterns = [
    path('test/', test),
    path('signUp/', signup),
    path('sendEmail/', sendEmail),
    path('sendEmail1/', sendEmailFP),
    path('FP/', FP),
    path('Query/', Query),
    path('login/', login),
    path('getUserDetails/', getUserDetails),
    path('mlModel/', mlModel),
    path('mlModel1/', mlModel1),
    path('mlModel2/', mlModel2),
]
