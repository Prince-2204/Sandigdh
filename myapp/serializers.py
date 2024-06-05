from rest_framework import serializers
from .models import *


# class ScreenshotSerializer(serializers.ModelSerializer):
    # class Meta:
    #     model = Screenshot
    #     fields = ['image']
        
# class FileUploadSerializer(serializers.Serializer):
#     file = serializers.FileField()
        



# ************New New***********

# class ScreenshotSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Screenshot
#         fields = ['image']
        
class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()


class ScreenshotSerializer(serializers.Serializer):
    screenshot = serializers.ImageField()
    class Meta:
        model = Screenshot
        fields = [ 'screenshot', 'id']


class UrlSerializer(serializers.ModelSerializer):
    class Meta:
        model = Url
        fields = ['url']

class UrlnewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Urlnew
        fields = ['url']



#******************from api wala*******************
class WebsiteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Website
        fields = ['url']
        
class CanonicSerializer(serializers.ModelSerializer):
    class Meta:
        model = Canonic
        fields = ['name']



class TagListSerializer(serializers.ModelSerializer):
    class Meta:
        model = TagList
        fields = ['id','strings']
