from django.urls import path
from .views import compare_faces

urlpatterns = [
    path('', compare_faces, name='compare_faces'),  # Root URL will now point to compare_faces view
]
