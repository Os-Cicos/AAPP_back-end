from django.urls import path
from . import views
from rest_framework_simplejwt import views as jwt_views

urlpatterns = [
     path('create_user/', 
          views.CreateUser.as_view(), 
          name='create_auth'),
     path('assistant/', 
          views.Assistant.as_view(),
          name='assistant'),
     path('loader/', 
          views.Loader.as_view(),
          name='Loader'),
     path('transcribe/', 
          views.Transcribe.as_view(),
          name='transcribe'),
     path('token/', 
          jwt_views.TokenObtainPairView.as_view(), 
          name ='token_obtain_pair'),
     path('token/refresh/', 
          jwt_views.TokenRefreshView.as_view(), 
          name ='token_refresh'),
]
