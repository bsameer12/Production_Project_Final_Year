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
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import get_user_model
from .utils import send_verification_email
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from .forms import ProfileForm, CustomPasswordChangeForm
from .models import Profile
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
from django.contrib import messages
from django.shortcuts import render, redirect


@login_required
def profile_view(request):
    user = request.user
    log_user_activity(request, action="Page Visit", description="Visited Profile page")

    # Ensure profile exists
    profile, _ = Profile.objects.get_or_create(user=user)

    # Bind our two forms
    profile_form = ProfileForm(
        request.POST or None,
        request.FILES or None,
        instance=profile,
        user=user
    )
    password_form = CustomPasswordChangeForm(user=user, data=request.POST or None)

    if request.method == 'POST':
        if 'update_profile' in request.POST:
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "Profile updated successfully!", extra_tags='profile')
                log_user_activity(request, action="Profile Update", description="Updated profile details")
                return redirect('profile')
            else:
                # field‐level errors will already be attached to profile_form
                for field, errs in profile_form.errors.items():
                    for err in errs:
                        messages.error(request, f"{field.capitalize()}: {err}", extra_tags='profile')

        elif 'change_password' in request.POST:
            if password_form.is_valid():
                password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, "Password changed successfully!", extra_tags='password')
                log_user_activity(request, action="Password Changed", description="User changed their password")
                return redirect('profile')
            else:
                for field, errs in password_form.errors.items():
                    for err in errs:
                        messages.error(request, f"{field.replace('_',' ').capitalize()}: {err}", extra_tags='password')

    return render(request, 'profile.html', {
        'profile_form': profile_form,
        'password_form': password_form,
    })


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
            # Pass all form errors to messages
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(
                        request,
                        f"{form.fields[field].label if field in form.fields else field.capitalize()}: {error}",
                        extra_tags='register'
                    )
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
                # Temporarily login user in session only to allow resend
                request.session['unverified_user_id'] = user.id
                return redirect('email_not_verified')
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


def email_not_verified_view(request):
    user_id = request.session.get('unverified_user_id')
    user = get_user_model().objects.filter(id=user_id).first()

    if not user or not hasattr(user, 'profile') or user.profile.is_verified:
        return redirect('login')  # fallback

    if request.method == 'POST':
        if 'resend' in request.POST:
            send_verification_email(user, request)
            messages.success(request, "Verification email resent. Please check your inbox.", extra_tags='email')
        elif 'logout' in request.POST:
            request.session.flush()
            return redirect('login')

    return render(request, 'email_not_verified.html', {'user': user})


@login_required
def admin_user_list_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Admin Users Management Page"
    )

    users = User.objects.exclude(id=request.user.id).select_related('profile').order_by('username')
    return render(request, 'admin_users.html', {'users': users})


@login_required
def delete_user_view(request, user_id):
    if not request.user.is_superuser:
        return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

    user = get_object_or_404(User, id=user_id)

    if user == request.user:
        return JsonResponse({'success': False, 'message': "You cannot delete yourself."}, status=400)

    if user.is_superuser:
        return JsonResponse({'success': False, 'message': "You cannot delete another admin."}, status=400)

    user.delete()

    log_user_activity(
        request,
        action="Delete",
        description=f"Admin deleted user: {user.username}"
    )

    return JsonResponse({'success': True, 'message': '✅ User deleted successfully.'})