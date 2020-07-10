from rest_framework import serializers,exceptions
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .models import signup
from django.contrib.auth import get_user_model
class loginSerializer(serializers.Serializer):
    username=serializers.CharField()
    password=serializers.CharField()

    def validate(self,data):
        username=data.get("username","")
        password=data.get("password","")
        print(data)
        if(username and password):
            user=authenticate(username=username,password=password)
            if user:
                if user.is_active:
                    data["user"]=user
                else:
                    msg="deactive"
                    raise exceptions.ValidationError(msg)
            else:
                msg="wrong"
                raise exceptions.ValidationError(msg)
        else:
            msg="invalid"
            raise exceptions.ValidationError(msg)
        return data
class UserSerializer(serializers.ModelSerializer):
    password=serializers.CharField(write_only=True)
    def create(self,validated_data):
        user=get_user_model().objects.create(
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user
    class Meta:
        model=get_user_model()
        fields=('username','password')

class signUpSerializer(serializers.ModelSerializer):
    image=serializers.ImageField(max_length=None,use_url=True)
    class Meta:
        model=signup
        fields=(
            'id',
            'phoneNumber',
            'name',
            'image',
            'registered_at',
            'dob',
            'gender','password',
            'verified',
            'otp'
        )