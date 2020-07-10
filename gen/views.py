from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from .models import *
from .serializers import *
from rest_framework import viewsets,generics,mixins
import math, random 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
from rest_framework.generics import CreateAPIView
from rest_framework.permissions import IsAuthenticated,AllowAny,IsAdminUser
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from django.contrib.auth import login as django_login,logout as django_logout
from rest_framework.authtoken.models import Token
from rest_framework.authentication import SessionAuthentication,BasicAuthentication,TokenAuthentication
from rest_framework.response import Response

#ML imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#buffer conversion
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import HttpResponse,JsonResponse
import base64
import PIL, PIL.Image
from io import StringIO
import io

d=os.path.dirname(os.getcwd())
d=os.path.join(d,"app")
d=os.path.join(d,"gen")
class MlModelIntegration(generics.GenericAPIView
    ,mixins.ListModelMixin
    ,mixins.CreateModelMixin
    ,mixins.RetrieveModelMixin
    ,mixins.UpdateModelMixin
    ,mixins.DestroyModelMixin):
    serializer_class=signUpSerializer
    queryset=signup.objects.all()
    lookup_field='id'
    authentication_classes=[TokenAuthentication]
    permission_classes=[IsAuthenticated,]

    def get(self,request):
        return JsonResponse({'a':'a'})
    def post(self,request):
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            x1=d
            # f=os.path.join(d,"States")
            e=os.path.join(d,request.data['state'])
            q=os.path.join(e,request.data['city'])
            # w=os.path.join(q,"jan")
            os.chdir(q)

            dataset = pd.read_csv(request.data['month']+'.csv')
            print(dataset)
            x=dataset.iloc[:,:-1].values 
            y=dataset.iloc[:,-1].values 
            print(x1)
            # os.chdir(x1)
            # l=os.path.join(x1,"2020")
            # j=os.path.join(l,"jan")
            os.chdir(x1)

            dataset2 = pd.read_csv(request.data['month']+'.csv')
            x2=dataset2.iloc[:,:-1].values
            #y2=dataset2.iloc[:,-1].values

            from sklearn.preprocessing import LabelEncoder,OneHotEncoder
            labelencoder=LabelEncoder()
            x[:,0]=labelencoder.fit_transform(x[:,0])
            onehotencoder=OneHotEncoder(categorical_features=[0])
            x=onehotencoder.fit_transform(x).toarray()
            labelencoder=LabelEncoder()
            x[:,-1]=labelencoder.fit_transform(x[:,-1])
            onehotencoder1=OneHotEncoder(categorical_features=[-1])
            x=onehotencoder1.fit_transform(x).toarray()
            x=x[:,1:]


            labelencoder=LabelEncoder()
            x2[:,0]=labelencoder.fit_transform(x2[:,0])
            # onehotencoder=OneHotEncoder(categorical_features=[0])
            onehotencoder=OneHotEncoder(categorical_features=[0])
            print(onehotencoder)
            x2=onehotencoder.fit_transform(x2).toarray()
            labelencoder=LabelEncoder()
            x2[:,-1]=labelencoder.fit_transform(x2[:,-1])
            onehotencoder=OneHotEncoder(categorical_features=[-1])
            x2=onehotencoder.fit_transform(x2).toarray()
            x2=x2[:,1:]
            '''
            from sklearn.model_selection import train_test_split
            x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)'''


            from sklearn.linear_model import LinearRegression
            regressor=LinearRegression()
            regressor.fit(x,y)

            y_pred=regressor.predict(x2)
            #plt.plot(y2,color='red',label='real')
            #plt.plot(y_pred,color='blue',label='pred')
            fig = plt.figure()
            plt.title('Cotton price') 
            plt.xlabel('time')
            #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

            x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            xi = list(range(len(x)))
            plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
            plt.xticks(xi, x)
            plt.legend
            # plt.show()
            # plt.close(fig)
            canvas = fig.canvas
            buf, size = canvas.print_to_buffer()
            image = PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
            buffer=io.BytesIO()
            image.save(buffer,'PNG')
            graphic = buffer.getvalue()
            graphic = base64.b64encode(graphic)
            buffer.close()
            a=str(graphic)
            a=a[a.find("'")+1:a.rfind("'")]
            print(request.user,request)
            return JsonResponse({"buffer":a})
        except Exception as e:
            return JsonResponse({"buffer":e})

class loginView(APIView):
    def post(self,request):
        
        serializer=loginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user=serializer.validated_data["user"]
        print(user,request,self)
        django_login(request,user)
        token,created=Token.objects.get_or_create(user=user)

        return Response({"token":token.key},status=200)



class logoutView(APIView):
    authentication_classes=(TokenAuthentication,)

    def post(self,request):
        django_logout(request)
        return Response(status=204)

class CreateUserView(CreateAPIView):
    model=get_user_model()
    permission_classes=(AllowAny,)
    serializer_class=UserSerializer

def sendMail(to,otp):
    fromaddr = "sihkkr2020@gmail.com"
    toaddr = to
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "SUBJECT OF THE MAIL"

    body = otp
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "demon_killers")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def generateOTP(): 
    digits = "0123456789"
    OTP = "" 
    for i in range(4) : 
        OTP += digits[math.floor(random.random() * 10)] 
  
    return OTP 

def test(request):
    return JsonResponse({'test':'pass'},status=200)
class GenAPI(generics.GenericAPIView
    ,mixins.ListModelMixin
    ,mixins.CreateModelMixin
    ,mixins.RetrieveModelMixin
    ,mixins.UpdateModelMixin
    ,mixins.DestroyModelMixin):
    serializer_class=signUpSerializer
    queryset=signup.objects.all()
    lookup_field='id'
    def post(self,request):
        try:
            print(request.data,self ,'anu')
            a=generateOTP()
            sendMail(request.data['id'],a)
            # request.data.otp=a
            request.data['otp']=a
            self.create(request)
            # print(request.data['email'],signup.objects.all().filter(email=request.data['email']))
            print(a)
            # self.update({'otp':a},request.data['email'])
            return JsonResponse({'otp':a})
        except Exception as e:
            return JsonResponse({'otp':e})
    def get(self,request):
        return self.list(request)
    def delete(self,request):
        # print((signup.objects.all().filter(pk=request.data['email'])[0].otp))
        # a=signup.objects.all().filter(pk=request.data['email'])[0].id
        signup.objects.all().filter(pk=request.data['id'])[0].delete()
        return JsonResponse({'delete':'success'})
class checkOtp(generics.GenericAPIView
    ,mixins.ListModelMixin
    ,mixins.CreateModelMixin
    ,mixins.RetrieveModelMixin
    ,mixins.UpdateModelMixin
    ,mixins.DestroyModelMixin):
    serializer_class=signUpSerializer
    queryset=signup.objects.all()
    lookup_field='id'
    def post(self,request):
        a=(signup.objects.all().filter(pk=request.data['id'])[0].otp)
        print(a,request.data['otp'])
        if(str(a)==(request.data['otp'])):
            m=signup.objects.all().filter(pk=request.data['id'])[0]
            m.verified='yes'
            m.save()
            
            requests.post('https://sih-django.herokuapp.com/gen/createUser/',{'username':m.id,'password':m.password})
            #making user
            return JsonResponse({'status':'correct otp'})
        else:
            signup.objects.all().filter(pk=request.data['id'])[0].delete()
            return JsonResponse({'status':'wrong otp'})


class GenAPI1(generics.GenericAPIView
    ,mixins.ListModelMixin
    ,mixins.CreateModelMixin
    ,mixins.RetrieveModelMixin
    ,mixins.UpdateModelMixin
    ,mixins.DestroyModelMixin):
    serializer_class=signUpSerializer
    queryset=signup.objects.all()
    lookup_field='id'
    def post(self,request,id=None):
        try:
            print(request.data,self ,'anu')
            a=generateOTP()
            sendMail(request.data['id'],a)
            # request.data.otp=a
            request.data['otp']=a
            # del req
            b=request.data['id']
            
            # del request.data['id']
            print(b,request.data)
            b=signup.objects.all().filter(pk=b)[0]
            b.otp=a
            b.save()
            # self.update(request)
            # print(request.data['email'],signup.objects.all().filter(email=request.data['email']))
            print(a)
            # self.update({'otp':a},b)
            return JsonResponse({'otp':a})
        except Exception as e:
            return JsonResponse({'otp':e})
class gu(generics.GenericAPIView
    ,mixins.ListModelMixin
    ,mixins.CreateModelMixin
    ,mixins.RetrieveModelMixin
    ,mixins.UpdateModelMixin
    ,mixins.DestroyModelMixin):
    serializer_class=signUpSerializer
    queryset=signup.objects.all()
    lookup_field='id'
    authentication_classes=[TokenAuthentication]
    permission_classes=[IsAuthenticated,]
    def post(self,request):
        try:
            print(request.user,request.user.id,request)
            # b=self.objects.all().filter(pk=str(request.user))[0]
            print((signup.objects.get(id=request.user)))
            d=dict()
            # b=[entry for entry in (signup.objects.get(id=request.user))]
            # print(list(queryset))
            print(signup.objects.get(id=request.user).image.url)
            b=signup.objects.get(id=request.user)
            d['name']=b.name
            d['phoneNumber']=b.phoneNumber
            d['id']=b.id
            d['registered_at']=b.registered_at
            d['dob']=b.dob
            d['image']=b.image.url
            d['gender']=b.gender
            d['password']=b.password
            d['verified']=b.verified
            d['otp']=b.otp
            return JsonResponse({'user':d})
        except Exception as e:
            return JsonResponse({'user':e})

