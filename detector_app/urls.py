from django.urls import path
from . import views

# Define your URL patterns here
urlpatterns = [
     path('start/', views.start_detection,name='start_detection'),
]