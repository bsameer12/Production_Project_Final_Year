import os
import json
import numpy as np
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from tensorflow.keras.models import load_model

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
@csrf_exempt
@login_required
def predict_landmarks(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            sequence = data.get("sequence")

            if not sequence or len(sequence) != 10:
                return JsonResponse({"error": "Invalid input. Sequence must contain 10 frames."}, status=400)

            sequence = np.array(sequence).reshape(1, 10, -1)
            preds = model.predict(sequence, verbose=0)[0]
            confidence = float(np.max(preds))
            label = class_labels[int(np.argmax(preds))]

            return JsonResponse({
                "label": label,
                "confidence": round(confidence, 2)
            })

        except Exception as e:
            return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Only POST requests are allowed."}, status=405)
