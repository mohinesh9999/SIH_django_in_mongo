from django.contrib import admin
# from django.urls import path,include
from rest_framework import routers
from django.conf.urls import include,url
from django.conf.urls.static import static
from django.conf import settings
from rest_framework.urlpatterns import format_suffix_patterns
from django.urls import path,include
from .views import *

urlpatterns = [
    path('test/',test),
    # path('signup/<str:id>/',GenAPI.as_view()),
    path('signup/',GenAPI.as_view()),
    path('getuser/',gu.as_view()),
    path('resend/',GenAPI1.as_view()),
    path('checkOtp/',checkOtp.as_view()),
    path('createUser/',CreateUserView.as_view()),
    path('mlmodel/',MlModelIntegration.as_view()),
    path('auth/login/',loginView.as_view()),
    path('auth/logout/',logoutView.as_view()),

]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)