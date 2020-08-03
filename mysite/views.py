from copy import copy
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
d=os.path.join(d,"sih")
xn=d
d=os.path.join(d,"States")



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
    z.append([q['msg'],q['name'],q['email'],datetime.datetime.now().isoformat()])
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
    global d,xn
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks')
        y=record.find_one({"_id":y['email']})
        z=y['recent']
        z.append([request.data['state'],request.data['city'],request.data['month'],datetime.datetime.now().isoformat()])
        print(y,z)
        record.update_many( {"_id":y['_id']}, { "$set":{  "recent":z} } ) 
        import matplotlib
        matplotlib.use('Agg')
        
        e=os.path.join(d,request.data['state'])
        q=os.path.join(e,request.data['city'].capitalize())
        os.chdir(q)
        mnth={'january': 'jan', 'february': 'feb', 'march': 'march', 
            'april': 'april', 'may': 'may', 'june': 'jun', 'july': 'july', 
            'august': 'august', 'september': 'sept', 'october': 'oct', 
            'november': 'nov', 'december': 'dec'}
        dataset = pd.read_csv(mnth[request.data['month']]+".csv")
        x=dataset.iloc[:,:-1].values 
        y=dataset.iloc[:,-1].values 


        size=y.size



        l=os.path.join(xn,"year")
        #j=os.path.join(l,"jan") 
        os.chdir(l)

        dataset2 = pd.read_csv(mnth[request.data['month']]+".csv")
        x2=dataset2.iloc[:,:-1].values
        #y2=dataset2.iloc[:,-1].values

        import math
        a=0
        yi=0
        for i in range(len(x)):
            a=a+x[i][2]
            yi=yi+x[i][3]   
        a/=size
        yi/=size



        for i in range(len(x2)):
            x2[i][2]=a
            x2[i][3]=yi

        x2

        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        from sklearn.compose import ColumnTransformer

        label_encoder_x_1 = LabelEncoder()
        x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')

        x

        label_encoder_x_2 = LabelEncoder()
        x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')


        #x=x[:,1:]




        label_encoder_x_2 = LabelEncoder()
        x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')

        label_encoder_x_2 = LabelEncoder()
        x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]             
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')
        #x2=x2[:,1:]




        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x,y)

        y_pred=regressor.predict(x2)
        #plt.plot(y2,color='red',label='real')
        #plt.plot(y_pred,color='blue',label='pred')
        plt.title('Cotton price') 
        plt.xlabel('time')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        xi = list(range(len(x)))
        plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
        plt.xticks(xi, x)
        plt.legend
        plt.show()
        l=[]
        for i in y_pred:
            l.append(i)
        return JsonResponse({"buffer":l,"flag":'True'})
    except Exception as e:
        return JsonResponse({"status": False,"flag":'False','e':e},status=400)
@api_view(['POST'])
def mlModel1(request):
    global d,xn
    try:
        y=request.data['token']
        y=jwt.decode(y, 'mks')
        import matplotlib
        matplotlib.use('Agg')
        
        e=os.path.join(d,request.data['state'])
        q=os.path.join(e,request.data['city'].capitalize())
        os.chdir(q)
        mnth={'january': 'jan', 'february': 'feb', 'march': 'march', 
            'april': 'april', 'may': 'may', 'june': 'jun', 'july': 'july', 
            'august': 'august', 'september': 'sept', 'october': 'oct', 
            'november': 'nov', 'december': 'dec'}
        dataset = pd.read_csv(mnth[request.data['month']]+".csv")
        x=dataset.iloc[:,:-1].values 
        y=dataset.iloc[:,-1].values 


        size=y.size



        l=os.path.join(xn,"year")
        #j=os.path.join(l,"jan") 
        os.chdir(l)

        dataset2 = pd.read_csv(mnth[request.data['month']]+".csv")
        x2=dataset2.iloc[:,:-1].values
        #y2=dataset2.iloc[:,-1].values

        import math
        a=0
        yi=0
        for i in range(len(x)):
            a=a+x[i][2]
            yi=yi+x[i][3]   
        a/=size
        yi/=size



        for i in range(len(x2)):
            x2[i][2]=a
            x2[i][3]=yi

        x2

        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        from sklearn.compose import ColumnTransformer

        label_encoder_x_1 = LabelEncoder()
        x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')

        x

        label_encoder_x_2 = LabelEncoder()
        x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')


        #x=x[:,1:]




        label_encoder_x_2 = LabelEncoder()
        x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')

        label_encoder_x_2 = LabelEncoder()
        x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]             
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')
        #x2=x2[:,1:]




        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x,y)

        y_pred=regressor.predict(x2)
        #plt.plot(y2,color='red',label='real')
        #plt.plot(y_pred,color='blue',label='pred')
        plt.title('Cotton price') 
        plt.xlabel('time')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        xi = list(range(len(x)))
        plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
        plt.xticks(xi, x)
        plt.legend
        plt.show()
        l=[]
        for i in y_pred:
            l.append(i)
        w=[]
        msp=5550
        for i in y_pred:
            if(msp>=i):
                w.append(msp)
            else:
                w.append(msp+0.4*(i-msp))
        return JsonResponse({"buffer":l,'w':w,'flag':'True'})
    except Exception as e:
        return JsonResponse({"status": False,"flag":'False','e':e},status=400)
@api_view(['POST'])
def mlModel2(request):
    global d,xn
    try:
        e=copy(d)
        y=request.data['token']
        y=jwt.decode(y, 'mks')
        import matplotlib
        matplotlib.use('Agg')
        e=os.path.join(d,request.data['state'])
        e=os.path.join(e,request.data['city'].capitalize())
        os.chdir(e)
        dataset = pd.read_csv('real.csv')
        # os.chdir(e)
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        indexedDataset = dataset.set_index(['Date'])
        indexedDataset = indexedDataset.fillna(method='ffill')


        from datetime import datetime
        #indexedDataset.tail(12)



        rolmean = indexedDataset.rolling(window=12).mean()

        rolstd = indexedDataset.rolling(window=12).std()
        #print(rolmean,rolstd)



        from statsmodels.tsa.stattools import adfuller

        #print('Results of DFT: ')
        dftest = adfuller(indexedDataset['Prices'],autolag='AIC')

        dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-val','lag used','Number of obser'])
        

        indexedDataset_logScale=np.log(indexedDataset)

        movingAverage=indexedDataset_logScale.rolling(window=12).mean()
        movingstd=indexedDataset_logScale.rolling(window=12).std()


        datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
        #datasetLogScaleMinusMovingAverage.head(12)

        datasetLogScaleMinusMovingAverage.dropna(inplace=True)
        #datasetLogScaleMinusMovingAverage.head(12)

        from statsmodels.tsa.stattools import adfuller
        def test_stationarity(timeseries):
            
            movingAverage=timeseries.rolling(window=12).mean()
            movingSTD=timeseries.rolling(window=12).std()
            
            dftest=adfuller(timeseries['Prices'],autolag='AIC')
            dfoutput=pd.Series(dftest[0:4],index=['Test stats','pval','lag','No of obser'])
        

        test_stationarity(datasetLogScaleMinusMovingAverage)

        exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()


        datasetLogScaleMinusMovingExponentialDecayAverage=indexedDataset_logScale-exponentialDecayWeightedAverage
        test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

        datasetLogDiffShifting=indexedDataset_logScale - indexedDataset_logScale.shift()


        datasetLogDiffShifting.dropna(inplace=True)
        test_stationarity(datasetLogDiffShifting)





        from statsmodels.tsa.arima_model import ARIMA

        model=ARIMA(indexedDataset_logScale,order=(1,1,1))
        results_AR=model.fit(disp=-1)


        predictions_ARIMA_diff=pd.Series(results_AR.fittedvalues,copy=True)
        #print(predictions_ARIMA_diff.head())

        predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
        #print(predictions_ARIMA_diff_cumsum.head())

        predictions_ARIMA_log=pd.Series(indexedDataset_logScale['Prices'].iloc[0],index=indexedDataset_logScale.index)
        predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        #predictions_ARIMA_log.head()

        predictions_ARIMA=np.exp(predictions_ARIMA_log)

        #indexedDataset_logScale
        #predictions_ARIMA

        modell=ARIMA(predictions_ARIMA,order=(1,1,1))
        results_ARM=modell.fit(disp=-1)

        results_ARM.plot_predict(1,60)
        x=results_ARM.forecast(steps=12)



        toplot=x[0][0:12]
        print(toplot,x[2])
        l=[]
        for i in toplot:
            l.append(i)
        w=[]
        for i in x[2]:
            i=list(i)
            w.append([i[0],i[1]])
        print(w)
        return JsonResponse({"buffer":l,'flag':'True',"x":w})
    except Exception as e:
        return JsonResponse({"status": False,"flag":'False','e':e},status=400)

import asyncio
l1=[]
def regression(city,state,month):
    try:
        global l1
        import matplotlib
        matplotlib.use('Agg')
        
        e=os.path.join(d,state)
        q=os.path.join(e,city.capitalize())
        os.chdir(q)
        mnth={'january': 'jan', 'february': 'feb', 'march': 'march', 
            'april': 'april', 'may': 'may', 'june': 'jun', 'july': 'july', 
            'august': 'august', 'september': 'sept', 'october': 'oct', 
            'november': 'nov', 'december': 'dec'}
        dataset = pd.read_csv(mnth[month]+".csv")
        x=dataset.iloc[:,:-1].values 
        y=dataset.iloc[:,-1].values 


        size=y.size



        l=os.path.join(xn,"year")
        #j=os.path.join(l,"jan") 
        os.chdir(l)

        dataset2 = pd.read_csv(mnth[month]+".csv")
        x2=dataset2.iloc[:,:-1].values
        #y2=dataset2.iloc[:,-1].values

        import math
        a=0
        yi=0
        for i in range(len(x)):
            a=a+x[i][2]
            yi=yi+x[i][3]   
        a/=size
        yi/=size



        for i in range(len(x2)):
            x2[i][2]=a
            x2[i][3]=yi

        x2

        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        from sklearn.compose import ColumnTransformer

        label_encoder_x_1 = LabelEncoder()
        x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')

        x

        label_encoder_x_2 = LabelEncoder()
        x[: , 1] = label_encoder_x_1.fit_transform(x[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]              
                )
            ],
            remainder='passthrough' 
        )
        x = transformer.fit_transform(x.tolist())
        x = x.astype('float64')


        #x=x[:,1:]




        label_encoder_x_2 = LabelEncoder()
        x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [0]              
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')

        label_encoder_x_2 = LabelEncoder()
        x2[: , 1] = label_encoder_x_1.fit_transform(x2[:,1])
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        
                OneHotEncoder(), 
                [1]             
                )
            ],
            remainder='passthrough' 
        )
        x2 = transformer.fit_transform(x2.tolist())
        x2 = x2.astype('float64')
        #x2=x2[:,1:]




        from sklearn.linear_model import LinearRegression
        regressor=LinearRegression()
        regressor.fit(x,y)

        y_pred=regressor.predict(x2)
        #plt.plot(y2,color='red',label='real')
        #plt.plot(y_pred,color='blue',label='pred')
        plt.title('Cotton price') 
        plt.xlabel('time')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        xi = list(range(len(x)))
        # plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
        # plt.xticks(xi, x)
        # plt.legend
        # plt.show()
        # l1.append(city,state,sum(list(y_pred))/(len(y_pred)))
        # import time
        # print(y_pred)
        # time.sleep(1)
        k=0
        for i in range(len(list(y_pred))):
            k+=float(y_pred[i])
        # print(city,state)
        # l1.append([city,state,(k/len(list(y_pred)))])
        # print('eeee')
        return [city,state,(k/len(list(y_pred)))]
    except Exception as e:
        print(e)
        pass
async def first():
    await asyncio.sleep(1)
    return "1"

async def second():
    await asyncio.sleep(1)
    return "2"
async def main():
    async def one_iteration():
        result = await first()
        print(result)
        result2 = await second()
        print(result2)
    coros = [one_iteration() for _ in range(12)]
    await asyncio.gather(*coros)
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
@api_view(['POST'])
def allprice(request):
    global l
    l=[]
    async def main():
        async def one_iteration(city,state,month):
            await regression(city,state,month)
        d={
                "andhrapradesh":["Kurnool"],
                "gujarat":["Ahmedabad","Amreli","Bhavnagar","Gandhinagar","Jamnagar","Junagarh","Kheda","Kutch","Rajkot"],
                "haryana":["Hisar","Jind","Sirsa"],
                "karnataka":["Bijapur","Davanagere","Dharwad","Haveri","Raichur"],
                "madhyapradesh":["Alrajpur","Badwani","Chindwara","Dhar","Khargone"],
                "maharashta":["Amravati","Aurangabad","Buldana","Nagpur","Nanded"],
                "punjab":["Barnala","Bathinda","Mansa"],
                "rajasthan":["Bhilwara","Hanumangarh","Sri Ganganagar"],
                "tamilnadu":["Tuticorin"],
                "telangana":["Khammam","Warangal"],
                "up":["Hathras" ]
            }
        x=[]
        x=[regression(j , i,'january') for j in d[i] for i in d]
        await asyncio.gather(*x)
    print('x')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print(l)
@api_view(['POST'])
def allprice1(request):
    global l1
    l=[]
    d={
                "andhrapradesh":["Kurnool"],
                "gujarat":["Ahmedabad","Amreli","Bhavnagar","Gandhinagar","Jamnagar","Junagarh","Kheda","Kutch","Rajkot"],
                "haryana":["Hisar","Jind","Sirsa"],
                "karnataka":["Bijapur","Davanagere","Dharwad","Haveri","Raichur"],
                "madhyapradesh":["Alrajpur","Badwani","Chindwara","Dhar","Khargone"],
                "maharashta":["Amravati","Aurangabad","Buldana","Nagpur","Nanded"],
                "punjab":["Barnala","Bathinda","Mansa"],
                "rajasthan":["Bhilwara","Hanumangarh","Sri Ganganagar"],
                "tamilnadu":["Tuticorin"],
                "telangana":["Khammam","Warangal"],
                "up":["Hathras" ]
            }
    x=[]
    # x=[regression(j , i,'january') for j in d[i] for i in d]
    from opencage.geocoder import OpenCageGeocode
    from geopy.distance import geodesic
    def find_distance(A,B):
        key = '710dc0082cfc492a8c90817f30efdff8'  # get api key from:  https://opencagedata.com
        geocoder = OpenCageGeocode(key)
        
        result_A = geocoder.geocode(A)
        lat_A = result_A[0]['geometry']['lat']
        lng_A = result_A[0]['geometry']['lng']
        
        result_B = geocoder.geocode(B)
        lat_B = result_B[0]['geometry']['lat']
        lng_B = result_B[0]['geometry']['lng']  
        
        return (geodesic((lat_A,lng_A), (lat_B,lng_B)).kilometers)
    for i in d:
        for j in d[i]:
            m=regression(j , i,request.data['month'])
            # w=find_distance('gwalior',j)
            av_price=7
            if(m):
                m.append(av_price*find_distance(request.data['city'],j))
                m.append(m[-1]+m[-2])
                x.append(m)
                # print()
            # print(i,j,m)av_price
    # print(l1)

     
    # def main():
    #     find_distance("kurukshetra","gwalior")
    # main()
    from copy import copy
    x1=copy(x)
    x.sort(key=lambda x:x[2])
    x1.sort(key=lambda x:x[4])
    return JsonResponse({"buffer":x,"buffer1":x1,'flag':'True'})