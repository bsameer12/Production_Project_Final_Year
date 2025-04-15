# accounts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile
from django.contrib.auth.forms import AuthenticationForm
class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    contact_number = forms.CharField(required=True, max_length=15)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'contact_number', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data["first_name"]
        user.last_name = self.cleaned_data["last_name"]
        user.email = self.cleaned_data["email"]

        if commit:
            user.save()

            # Save profile fields
            profile, created = Profile.objects.get_or_create(user=user)
            profile.contact_number = self.cleaned_data["contact_number"]
            profile.save()

        return user


class CustomLoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'placeholder': 'Username',
            'autocomplete': 'username',
            'class': 'with-placeholder'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'placeholder': 'Password',
            'autocomplete': 'current-password',
            'class': 'with-placeholder'
        })
    )