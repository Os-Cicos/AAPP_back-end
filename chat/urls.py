from django.urls import path
from . import views
from rest_framework_simplejwt import views as jwt_views

urlpatterns = [
     path('assistant/', views.Assistant.as_view(),name='assistant'),
     path('loader/', views.Loader.as_view(),name='Loader'),
     path('transcribe/', views.Transcribe.as_view(),name='transcribe'),
]
