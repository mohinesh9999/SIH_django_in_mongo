from pymongo import MongoClient  #for mongodb
import jwt    #for authentication
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_protect,csrf_exempt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import math,random
from django.core.mail import send_mail
import datetime;
#connecting python with mongodb
client=MongoClient("mongodb+srv://test:test@cluster0-nc9ml.mongodb.net/sih?retryWrites=true&w=majority")
db=client.get_database('sih')
record=db.sih
from rest_framework.decorators import api_view
import hashlib 
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
# d=os.path.join(d,"mysite")
d=os.path.join(d,"app")
d=os.path.join(d,"gen")



def test(request):
    return JsonResponse({'test':'pass'},status=200)
def generateOTP(): 
    digits = "0123456789"
    OTP = "" 
    for i in range(4) : 
        OTP += digits[math.floor(random.random() * 10)] 
    return OTP
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
@api_view(['POST'])
def signup(request):
    # print((request))
    if request.method == "POST":
        try:
            # print((request),request.data)
            x=record.insert_one(dict(request.data))
            return JsonResponse({"status": dict(request.data)},status=200)
        except Exception as e:
            return JsonResponse({"status": "email already exist"},status=500)
    else:
        return JsonResponse({"status": "Only post method allowed"},status=500)

@api_view(['POST'])
def sendEmail(request):
    q=request.data
    x=record.find_one({"_id":q['email']})
    if(x==None):
        try:
            otp=generateOTP()
            print(request.data['email'],otp)
            # sendEmail(request.data['email'],otp)
            send_mail('Verificaton for signup', otp, 'sihkkr2020@gmail.com', [request.data['email']])
            return JsonResponse({"status":hashlib.md5(otp.encode()).hexdigest()},status=200)
        except Exception as e:
            return JsonResponse({"status": "an error occured :(","e":e},status=500)
    else:
        return JsonResponse({"status":'already registered'},status=200)


@api_view(['POST'])
def sendEmailFP(request):
    q=request.data
    x=record.find_one({"_id":q['email']})
    if(x!=None):
        try:
            otp=generateOTP()
            print(request.data['email'],otp)
            # sendEmail(request.data['email'],otp)
            send_mail('Verificaton for signup', otp, 'sihkkr2020@gmail.com', [request.data['email']])
            return JsonResponse({"status":hashlib.md5(otp.encode()).hexdigest()},status=200)
        except Exception as e:
            return JsonResponse({"status": "an error occured :(","e":e},status=500)
    else:
        return JsonResponse({"status":'not registerd'},status=200)


@api_view(['POST'])
def FP(request):
    q=request.data
    record.update_many( {"_id":q['email']}, { "$set":{  "password":q['password']} } ) 
    return JsonResponse({"status":'done'},status=200)



@api_view(['POST'])
def Query(request):
    q=request.data
    y=request.data['token']
    y=jwt.decode(y, 'mks')
    y1=record.find_one({"_id":y['email']})
    z=y1['query']
    z.append([q['msg'],q['name'],q['email'],datetime.datetime.now().timestamp()])
    record.update_many( {"_id":y['email']}, { "$set":{  "query":z} } ) 
    return JsonResponse({"status":'done'},status=200)





@api_view(['POST'])
def login(request):
    try:
        q=request.data
        print(q)
        y=jwt.encode({"email":q['email']},"mks")
        x=record.find_one({"_id":q['email'],"password":q['password']})
        print(y.decode('UTF-8'),x,q,jwt.decode(y.decode('UTF-8'), 'mks'))
        if(x!=None):
            return JsonResponse({"status": "True","token":y.decode('UTF-8')},status=200)
        else:
            return JsonResponse({"status": "False"},status=200)
    except Exception as e:
        return JsonResponse({"status": "an error occured :(","e":e},status=500)











@api_view(['POST'])
def getUserDetails(request):
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks')
        y=record.find_one({"_id":y['email']})
        return JsonResponse({"status": "True","details":y},status=200)
    except Exception as e:
        return JsonResponse({"status": "an error occured :(","e":e},status=500)








@api_view(['POST'])
def mlModel(request):
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks')
        y=record.find_one({"_id":y['email']})
        z=y['recent']
        z.append([request.data['state'],request.data['city'],request.data['month'],datetime.datetime.now().timestamp()])
        print(y,z)
        record.update_many( {"_id":y['_id']}, { "$set":{  "recent":z} } ) 
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
        print(type(y_pred))
        l=[]
        for i in y_pred:
            l.append(i)
        return JsonResponse({"buffer":l})
    except Exception as e:
        return JsonResponse({"status": "an error occured :(","e":e},status=500)