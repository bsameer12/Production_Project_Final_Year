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

@login_required
def profile_view(request):
    user = request.user

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

        profile.save()  # ✅ don't skip this

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
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                if user.profile.is_verified:
                    login(request, user)
                    messages.success(request, f'Welcome back, {user.username}!')
                    return redirect('predict')  # Update this to your main page
                else:
                    messages.error(request, "Please verify your email before logging in.")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid login details.")
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')


def verify_email(request, token):
    profile = get_object_or_404(Profile, email_token=token)
    if not profile.is_verified:
        profile.is_verified = True
        profile.save()
        messages.success(request, 'Email verified successfully!')
    else:
        messages.info(request, 'Email already verified.')
    return redirect('login')