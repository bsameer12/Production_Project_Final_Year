from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('predict/', views.predict_dashboard, name='predict'),
    path("predict_landmarks/", views.predict_landmarks, name="predict_landmarks"),  # ✅ AJAX endpoint
    path('prediction-history/', views.prediction_history_view, name='prediction_history'),
    path('user-history/', views.user_history_view, name='user_history'),
    path('admin_prediction-history/', views.admin_prediction_history_view, name='admin_prediction_history'),
    path('admin-user-history/', views.admin_user_history_view, name='admin_user_history'),
    path('generate_sentence/', views.generate_sentence_view, name='generate_sentence'),
    path('sentence-history/', views.sentence_history_view, name='sentence_history'),
    path('admin_sentence-history/', views.admin_sentence_history_view, name='admin_sentence_history'),
    path("english-to-asl/", views.english_to_asl_view, name="english_to_asl"),
    path("generate-asl-video/", views.generate_asl_video, name="generate_asl_video"),
    path('asl-video-history/', views.asl_video_history, name='asl_video_history'),
    path('admin-video-history/', views.admin_video_history_view, name='admin_video_history'),







]


