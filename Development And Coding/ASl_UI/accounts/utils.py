from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from asl.models import AuditLog
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.urls import reverse
from django.conf import settings

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

    subject = 'Verify Your Email - ASL Sign Translator'

    # Context for rendering the template
    context = {
        'user': user,
        'verification_link': link,
    }

    # Render HTML content from template
    html_content = render_to_string('emails/verify_email.html', context)

    # Fallback plain text message
    text_content = f"""
Hello {user.get_full_name() or user.username},

Please verify your email address for your ASL Sign Translator account.

Click the link below:
{link}

If you didn’t request this, you can ignore this email.

— ASL Sign Translator Support Team
"""

    # Create the email
    email = EmailMultiAlternatives(
        subject=subject,
        body=text_content,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=[user.email],
    )
    email.attach_alternative(html_content, "text/html")
    email.send(fail_silently=False)
