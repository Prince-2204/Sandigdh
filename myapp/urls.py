# urls.py
from django.urls import path
from .views import *
from . import views

urlpatterns = [
    path('',views.new,name="new"),
    # 'upload-screenshot/' API is to upload the screenshot in the database
    path('upload-screenshot/', UploadScreenshot.as_view(), name='upload_screenshot'),
    #   'fetch-url/' is to  
    path('fetch-url/', views.FetchURLView.as_view(), name='fetch_url'),
    # 'api/receive-url/' API is to fetch the URL of the website containing fake urgency 
    path('api/receive-url/', views.receive_url, name='receive_url'),
    # 'api/receive-urlnew/' API is to fetch the URL of the website containing fake review
    path('api/receive-urlnew/', views.receive_url_new, name='receive_url'),
    #'testing' API is to test the code if there is any bug in it.
    path('testing', views.admin_redirect, name='testing'),
    # 'api/urgencystring/' API is used to send the information regarding the fake urgency
    path('api/urgencystring/', SendStringdiscountAPIView.as_view(), name='string-list'),
    #'api/maliciousstring/' API is used to send the information related to malicious link present in any
    # commercial website
    path('api/maliciousstring/', SendMaliciousStringAPIView.as_view(), name='malicious-list'),
    #'api/websites/' API is to fetch the URL of the current website you are visiting
    path('api/websites/', WebsiteView.as_view(), name='websites'),
    #'api/send-string/' API is to Send the information related to Percentage of Dark Pattern in the website 
    #you are visiting
    path('api/send-string/', SendStringAPIView.as_view(), name='send-string'),
    #'api/tags/' API is to send the elements tag name which contains the dark pattern
    path('api/tags/', TagListView.as_view(), name='string-list'),
    #'api/forced-string/' API is to send the information regarding the forced account creation Dark Pattern
    path('api/forced-string/', SendforcedStringAPIView.as_view(), name='forced-string'),
    #'api/fakereviewstring/' API is to send the Fake review data to extension
    path('api/fakereviewstring/', SendStringfakereviewAPIView.as_view(), name='string-list'),
    # 'api/detailedreport/' API is to get the detailed analysis of all types of dark pattern on single click
    path('api/detailedreport/', DetailedListView.as_view(), name='string-list'),

]
