from django.urls import path
from . import views
from rest_framework_simplejwt import views as jwt_views

urlpatterns = [
     path("", views.APIRootView.as_view(), name="api-root"),
     path('assistant/', views.assistant.as_view(), name='assistant'),
     path('token/', 
          jwt_views.TokenObtainPairView.as_view(), 
          name ='token_obtain_pair'),
     path('token/refresh/', 
          jwt_views.TokenRefreshView.as_view(), 
          name ='token_refresh'),
]