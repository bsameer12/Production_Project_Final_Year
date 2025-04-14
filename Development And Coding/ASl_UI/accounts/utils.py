from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse


def send_verification_email(user, request):
    token = user.profile.email_token
    link = request.build_absolute_uri(reverse('verify_email', kwargs={'token': token}))
    subject = 'Verify Your Email'
    message = f'Hello {user.username},\n\nClick the link below to verify your email:\n{link}'

    send_mail(
        subject,
        message,
        settings.DEFAULT_FROM_EMAIL,
        [user.email],
        fail_silently=False,
    )
