from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from .views import email_not_verified_view
from .views import admin_user_list_view, delete_user_view



urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('verify/<uuid:token>/', views.verify_email, name='verify_email'),
    path('profile/', views.profile_view, name='profile'),
    path('email-not-verified/', email_not_verified_view, name='email_not_verified'),
    path('admin/users/', admin_user_list_view, name='admin_user_list'),
    path('admin/users/delete/<int:user_id>/', delete_user_view, name='delete_user'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
