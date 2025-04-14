import os
import json
import numpy as np
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from .models import ASLPrediction
from django.utils.timezone import now
from .utils import log_user_activity
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import ASLPrediction
from .models import AuditLog
from django.core.exceptions import PermissionDenied

@login_required
def prediction_history_view(request):
    predictions = ASLPrediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'prediction_history.html', {'predictions': predictions})


# === Load your model once globally ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'asl_landmark_lstm_model.keras')
model = load_model(MODEL_PATH)

# === Label map for predictions ===
label_map = [
    '1', '10', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
class_labels = {i: label for i, label in enumerate(label_map)}

# === Dashboard View ===
@login_required
def predict_dashboard(request):
    return render(request, "dashboard.html", {"user": request.user})




# === AJAX Prediction Endpoint ===
# === AJAX Prediction Endpoint ===
@csrf_exempt
@login_required
def predict_landmarks(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            sequence = data.get("sequence")

            if not sequence or len(sequence) != 10:
                return JsonResponse({"error": "Invalid input. Sequence must contain 10 frames."}, status=400)

            sequence_np = np.array(sequence).reshape(1, 10, -1)
            preds = model.predict(sequence_np, verbose=0)[0]

            top3_indices = preds.argsort()[-3:][::-1]
            top3 = [{"label": class_labels[i], "confidence": round(float(preds[i]), 2)} for i in top3_indices]

            # Save to DB
            ASLPrediction.objects.create(
                user=request.user,
                input_sequence=sequence,
                top1_label=top3[0]["label"],
                top1_confidence=top3[0]["confidence"],
                top2_label=top3[1]["label"],
                top2_confidence=top3[1]["confidence"],
                top3_label=top3[2]["label"],
                top3_confidence=top3[2]["confidence"],
            )

            # After successful prediction
            log_user_activity(
                request,
                action="Prediction",
                description=f"Top 1: {top3[0]['label']} ({top3[0]['confidence']}), "
                            f"Top 2: {top3[1]['label']} ({top3[1]['confidence']}), "
                            f"Top 3: {top3[2]['label']} ({top3[2]['confidence']})"
            )

            return JsonResponse({
                "label": top3[0]["label"],
                "confidence": top3[0]["confidence"],
                "top2": top3[1],
                "top3": top3[2]
            })

        except Exception as e:
            return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Only POST requests are allowed."}, status=405)



@login_required
def user_history_view(request):
    audit_logs = AuditLog.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'user_history.html', {'predictions': audit_logs})


@login_required
def admin_prediction_history_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    predictions = ASLPrediction.objects.exclude(user=request.user).select_related('user').order_by('-created_at')

    return render(request, 'admin_prediction_history.html', {'predictions': predictions})


@login_required
def admin_user_history_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    audit_logs = AuditLog.objects.exclude(user=request.user).select_related('user').order_by('-timestamp')
    return render(request, 'user_history.html', {'predictions': audit_logs})