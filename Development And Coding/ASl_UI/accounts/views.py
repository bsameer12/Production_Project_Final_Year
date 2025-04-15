from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from .forms import CustomUserCreationForm
from .utils import send_verification_email
from django.shortcuts import get_object_or_404
from .models import Profile
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .models import Profile
from .forms import CustomLoginForm  # Your form with placeholders
from django.contrib.auth.views import (
    PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
)
from django.urls import reverse_lazy
from asl.models import AuditLog
from asl.utils import log_user_activity  # If stored in a utils.py file



@login_required
def profile_view(request):
    user = request.user

    # 📘 Log user activity: page visit
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Profile page"
    )

    # Ensure profile exists
    if not hasattr(user, 'profile'):
        Profile.objects.create(user=user)

    profile = user.profile

    if request.method == 'POST':
        user.first_name = request.POST.get('first_name')
        user.last_name = request.POST.get('last_name')
        user.save()

        profile.contact_number = request.POST.get('contact')

        if request.FILES.get('profile_picture'):
            profile.profile_picture = request.FILES.get('profile_picture')
            print("✅ Received profile_picture:", profile.profile_picture)

        profile.save()

        # ✅ Log profile update action
        log_user_activity(
            request,
            action="Profile Update",
            description="Updated profile details"
        )

        return redirect('profile')

    return render(request, 'profile.html', {'user': user, 'profile': profile})


def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            send_verification_email(user, request)
            messages.success(request, 'Registered successfully. Check your email to verify your account.')
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('predict')  # Already logged in

    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if hasattr(user, 'profile') and user.profile.is_verified:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect('predict')
            else:
                messages.error(request, "Please verify your email before logging in.")
        else:
            messages.error(request, "Please enter a correct username and password.")
    else:
        form = CustomLoginForm()

    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')


def verify_email(request, token):
    profile = get_object_or_404(Profile, email_token=token)

    # Use the associated user from the profile for logging
    user = profile.user

    if not profile.is_verified:
        profile.is_verified = True
        profile.save()
        messages.success(request, 'Email verified successfully!')

        # 🟢 Log the successful verification
        log_user_activity(
            request,
            action="Email Verification",
            description=f"User '{user.username}' verified their email successfully."
        )
    else:
        messages.info(request, 'Email already verified.')

        # 🔵 Log the already verified case
        log_user_activity(
            request,
            action="Email Verification",
            description=f"User '{user.username}' attempted to verify an already verified email."
        )

    return redirect('login')

class CustomPasswordResetView(PasswordResetView):
    template_name = 'auth/password_reset.html'
    email_template_name = 'auth/password_reset_email.txt'
    html_email_template_name = 'auth/password_reset_email.html'
    subject_template_name = 'auth/password_reset_email.txt'
    success_url = reverse_lazy('password_reset_done')

    def form_valid(self, form):
        # 🔍 Try to find the user for audit logging
        email = form.cleaned_data.get('email')
        user = User.objects.filter(email=email).first()

        if user:
            log_user_activity(
                self.request,
                action="Password Reset Requested",
                description=f"Password reset link requested for user '{user.username}'"
            )
        else:
            log_user_activity(
                self.request,
                action="Password Reset Attempt",
                description=f"Password reset attempted for non-registered email: {email}"
            )

        return super().form_valid(form)

class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'auth/password_reset_done.html'

    def get(self, request, *args, **kwargs):
        # 📝 Log password reset confirmation page visit
        log_user_activity(
            request,
            action="Password Reset Email Sent",
            description="Visited Password Reset Done page after submitting email"
        )
        return super().get(request, *args, **kwargs)
    
class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'auth/password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')

class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'auth/password_reset_complete.html'