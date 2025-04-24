from django import forms
from django.contrib.auth.forms import (
    UserCreationForm,
    AuthenticationForm,
    PasswordChangeForm
)
from django.contrib.auth import get_user_model
from .models import Profile

User = get_user_model()

class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'First name', 'class': 'with-placeholder'})
    )
    last_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'Last name', 'class': 'with-placeholder'})
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'placeholder': 'Email address', 'class': 'with-placeholder'})
    )
    contact_number = forms.RegexField(
        regex=r'^\+?\d{7,15}$',
        required=True,
        error_messages={
            'invalid': 'Enter a valid phone number (7–15 digits, optional +).'
        },
        widget=forms.TextInput(attrs={'placeholder': '+977XXXXXXXXXX', 'class': 'with-placeholder'})
    )

    class Meta:
        model = User
        fields = [
            'username',
            'first_name',
            'last_name',
            'email',
            'contact_number',
            'password1',
            'password2',
        ]

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            profile, _ = Profile.objects.get_or_create(user=user)
            profile.contact_number = self.cleaned_data['contact_number']
            profile.save()
        return user

class ProfileForm(forms.ModelForm):
    first_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'First name', 'class': 'with-placeholder'})
    )
    last_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'Last name', 'class': 'with-placeholder'})
    )
    contact_number = forms.RegexField(
        regex=r'^\+?\d{7,15}$',
        required=True,
        error_messages={
            'invalid': 'Enter a valid phone number (7–15 digits, optional +).'
        },
        widget=forms.TextInput(attrs={'placeholder': '+977XXXXXXXXXX', 'class': 'with-placeholder'})
    )

    class Meta:
        model = Profile
        fields = ['first_name', 'last_name', 'contact_number', 'profile_picture']

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user')
        super().__init__(*args, **kwargs)
        self.user = user
        self.fields['first_name'].initial = user.first_name
        self.fields['last_name'].initial = user.last_name

    def save(self, commit=True):
        profile = super().save(commit=False)
        self.user.first_name = self.cleaned_data['first_name']
        self.user.last_name = self.cleaned_data['last_name']
        if commit:
            self.user.save()
            profile.user = self.user
            profile.save()
        return profile

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

class CustomPasswordChangeForm(PasswordChangeForm):
    old_password = forms.CharField(
        label="Current password",
        strip=False,
        widget=forms.PasswordInput(attrs={
            'autocomplete': 'current-password',
            'placeholder': 'Current password',
            'class': 'with-placeholder'
        })
    )
    new_password1 = forms.CharField(
        label="New password",
        strip=False,
        widget=forms.PasswordInput(attrs={
            'autocomplete': 'new-password',
            'placeholder': 'New password',
            'class': 'with-placeholder'
        }),
        help_text=PasswordChangeForm.base_fields['new_password1'].help_text
    )
    new_password2 = forms.CharField(
        label="Confirm new password",
        strip=False,
        widget=forms.PasswordInput(attrs={
            'autocomplete': 'new-password',
            'placeholder': 'Confirm new password',
            'class': 'with-placeholder'
        })
    )

    # No __init__ override needed; PasswordChangeForm's constructor accepts the 'user' argument
