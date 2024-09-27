from django.urls import path
from .views import compare_faces, about  # Import the new view

urlpatterns = [
    path('', compare_faces, name='compare_faces'),  # Root URL
    path('about/', about, name='about'),  # URL for the About page
]
