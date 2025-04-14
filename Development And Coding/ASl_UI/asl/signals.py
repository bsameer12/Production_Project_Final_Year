from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.dispatch import receiver
from .utils import log_user_activity

@receiver(user_logged_in)
def on_user_login(sender, request, user, **kwargs):
    log_user_activity(request, "Login", "User logged in.")

@receiver(user_logged_out)
def on_user_logout(sender, request, user, **kwargs):
    log_user_activity(request, "Logout", "User logged out.")
