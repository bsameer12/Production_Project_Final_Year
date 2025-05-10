from .models import AuditLog

def log_user_activity(request, action, description):
    ip = get_client_ip(request)
    agent = request.META.get('HTTP_USER_AGENT', '')
    user = request.user if request.user.is_authenticated else None

    AuditLog.objects.create(
        user=user,
        action=action,
        description=description,
        ip_address=ip,
        user_agent=agent
    )

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0]
    return request.META.get('REMOTE_ADDR')
