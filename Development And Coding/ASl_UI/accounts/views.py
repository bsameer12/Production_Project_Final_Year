from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.urls import reverse_lazy
from django.contrib.auth.views import (
    PasswordResetView, PasswordResetDoneView,
    PasswordResetConfirmView, PasswordResetCompleteView
)

from .models import Profile
from .forms import CustomUserCreationForm, CustomLoginForm
from .utils import send_verification_email
from asl.utils import log_user_activity
from asl.models import AuditLog


@login_required
def profile_view(request):
    user = request.user

    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Profile page"
    )

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

        profile.save()

        messages.success(request, "Profile updated successfully!", extra_tags='profile')

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
            messages.success(
                request,
                'Registered successfully. Check your email to verify your account.',
                extra_tags='register'
            )
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('predict')

    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if hasattr(user, 'profile') and user.profile.is_verified:
                login(request, user)
                name = user.first_name.strip() if user.first_name.strip() else user.username
                messages.success(request, f"Welcome back, {name}!", extra_tags='login')
                return redirect('predict')
            else:
                messages.error(request, "Please verify your email before logging in.", extra_tags='login')
        else:
            messages.error(request, "Please enter a correct username and password.", extra_tags='login')
    else:
        form = CustomLoginForm()

    return render(request, 'login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


def verify_email(request, token):
    profile = get_object_or_404(Profile, email_token=token)
    user = profile.user

    if not profile.is_verified:
        profile.is_verified = True
        profile.save()
        messages.success(request, 'Email verified successfully!', extra_tags='login')
        log_user_activity(
            request,
            action="Email Verification",
            description=f"User '{user.username}' verified their email successfully."
        )
    else:
        messages.info(request, 'Email already verified.', extra_tags='login')
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
        log_user_activity(
            request,
            action="Password Reset Email Sent",
            description="Visited Password Reset Done page after submitting email"
        )
        return super().get(request, *args, **kwargs)


class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'auth/password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')

    def get(self, request, *args, **kwargs):
        log_user_activity(
            request,
            action="Password Reset Link Accessed",
            description="Visited Password Reset Confirm page to enter new password"
        )
        return super().get(request, *args, **kwargs)


class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'auth/password_reset_complete.html'

    def get(self, request, *args, **kwargs):
        log_user_activity(
            request,
            action="Password Reset Complete",
            description="User landed on Password Reset Complete page after setting a new password"
        )
        messages.success(request, "Your password has been reset successfully!", extra_tags='password')
        return super().get(request, *args, **kwargs)
