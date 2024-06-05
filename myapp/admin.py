from django.contrib import admin
from .models import *

# Register your models here.

admin.site.register(Screenshot)
admin.site.register(Url)

#**************from api wala************

admin.site.register(Website)
admin.site.register(Canonic)
admin.site.register(TagList)
admin.site.register(Urlnew)
