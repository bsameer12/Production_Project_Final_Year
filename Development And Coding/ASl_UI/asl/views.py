from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def predict_dashboard(request):
    return render(request, "dashboard.html", {"user": request.user})
