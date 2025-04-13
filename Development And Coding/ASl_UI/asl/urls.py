from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_dashboard, name='predict'),
    path("predict_landmarks/", views.predict_landmarks, name="predict_landmarks"),  # ✅ AJAX endpoint
]

