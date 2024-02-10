from django.db import models

class User(models.Model):
    idUser = models.CharField(max_length=100, unique=True)
    start_time = models.DateTimeField(auto_now_add=True)
    count = models.IntegerField(default=0)

    def __str__(self):
        return f"idUser: {self.idUser}"
