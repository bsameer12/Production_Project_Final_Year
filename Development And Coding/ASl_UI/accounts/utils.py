from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from asl.models import AuditLog

def log_user_activity(request, action, description):
    try:
        AuditLog.objects.create(
            user=request.user if request.user.is_authenticated else None,
            action=action,
            description=description,
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', 'Unknown')
        )
    except Exception as e:
        print(f"[AuditLog Error] {e}")

def get_client_ip(request):
    """Safely gets the real client IP address, considering proxy headers."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR', '')
    return ip



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
