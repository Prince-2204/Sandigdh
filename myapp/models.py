from django.db import models






# ****************NEW NEW*******************
    
class Screenshot(models.Model):
    image = models.ImageField(upload_to='screenshots/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Url(models.Model):
    url = models.TextField()
    # created_at = models.DateTimeField(auto_now_add=True)

    # def __str__(self):
    #     return self.url


class Urlnew(models.Model):
    url = models.TextField()

#*********from api wala************
class Website(models.Model):
    url = models.URLField()

class Canonic(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name
    

class TagList(models.Model):
    tags = models.TextField()