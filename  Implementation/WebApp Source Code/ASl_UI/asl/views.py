from tensorflow.keras.models import load_model
from .utils import log_user_activity
from django.shortcuts import render
from .models import ASLPrediction
from .models import AuditLog
from django.core.exceptions import PermissionDenied
import google.generativeai as genai
import json
from .models import ASLSentenceGeneration
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils.timezone import now
import os
import cv2
import numpy as np
import imageio
from django.conf import settings
from .models import ASLVideoHistory
import logging
from django.shortcuts import get_object_or_404


logger = logging.getLogger(__name__)

@csrf_exempt
@login_required
def generate_sentence_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            predictions = body.get("predictions", [])
            print("🧠 Received predictions:", predictions)

            if not predictions:
                return JsonResponse({"error": "No predictions provided."}, status=400)

            # 🌐 Gemini Config
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

            is_letters = all(len(p) == 1 for p in predictions)
            combined = "".join(predictions) if is_letters else " ".join(predictions)

            prompt = (
                f"What could the following signed letters mean? {combined}. "
                f"Suggest the most likely English word or greeting."
                if is_letters and len(predictions) <= 2
                else f"The following sequence of signed inputs was detected: {combined}. "
                     f"Convert this into a meaningful and grammatically correct English sentence."
            )

            # 🔮 Gemini Prediction
            response = model.generate_content(prompt)
            sentence = response.text.strip() if hasattr(response, 'text') else "No response generated."

            print("✅ Gemini Response:", sentence)

            # 💾 Save to DB
            ASLSentenceGeneration.objects.create(
                user=request.user,
                predictions=predictions,
                generated_sentence=sentence
            )

            # 🕵️ Log user activity
            log_user_activity(
                request,
                action="Sentence Generation",
                description=f"Input: {combined} → Output: {sentence}"
            )

            return JsonResponse({"sentence": sentence})

        except Exception as e:
            print("❌ Gemini backend error:", str(e))
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "GET not allowed"}, status=405)



@login_required
def prediction_history_view(request):
    # 📘 Log user activity
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Prediction History page"
    )

    # 📄 Fetch predictions
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
    # 📝 Audit log for dashboard visit
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited ASL Prediction Dashboard"
    )

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
    # 📝 Log this page visit
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited User Audit History"
    )

    audit_logs = AuditLog.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'user_history.html', {'predictions': audit_logs})


@login_required
def admin_prediction_history_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    # 📝 Log audit trail for viewing others' prediction history
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Users Prediction History (Admin View)"
    )

    predictions = ASLPrediction.objects.exclude(user=request.user).select_related('user').order_by('-created_at')
    return render(request, 'admin_prediction_history.html', {'predictions': predictions})

@login_required
def admin_user_history_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    # 📝 Audit log entry for admin viewing others' audit trail
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Users Audit History (Admin View)"
    )

    audit_logs = AuditLog.objects.exclude(user=request.user).select_related('user').order_by('-timestamp')
    return render(request, 'admin_user_history.html', {'predictions': audit_logs})


@login_required
def sentence_history_view(request):
    # 📝 Log the visit to sentence history page
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited Sentence Generation History"
    )

    generations = ASLSentenceGeneration.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'sentence_history.html', {'generations': generations})

@login_required
def admin_sentence_history_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    # ✍️ Audit log for admin viewing others' sentence generation
    log_user_activity(
        request,
        action="Page Visit",
        description="Admin visited all users' Sentence Generation History"
    )

    sentences = ASLSentenceGeneration.objects.exclude(user=request.user).select_related('user').order_by('-created_at')

    return render(request, 'admin_sentence_history.html', {'sentences': sentences})



@csrf_exempt
@login_required
def generate_asl_video(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid method"}, status=405)

    text = request.POST.get('text', '').upper().strip()
    if not text:
        return JsonResponse({"error": "No input provided."}, status=400)

    images_path = os.path.join(settings.BASE_DIR, 'asl_images')
    output_dir = os.path.join(settings.MEDIA_ROOT, 'asl_videos')
    os.makedirs(output_dir, exist_ok=True)

    fps = 2
    hold_time = 1
    frames_per_char = fps * hold_time
    frames = []

    base_width = base_height = None

    for char in text:
        filename = "space.png" if char == " " else f"{char}.png"
        img_path = os.path.join(images_path, filename)

        if not os.path.exists(img_path):
            img_path = img_path.replace('.png', '.jpg')

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                continue

            if base_width is None or base_height is None:
                base_height, base_width = img.shape[:2]
            else:
                img = cv2.resize(img, (base_width, base_height))

            overlay_text = f"{request.user.username.upper()} | {now().strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(img, overlay_text, (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for _ in range(frames_per_char):
                frames.append(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))

    if not frames:
        return JsonResponse({"error": "No valid ASL characters found."}, status=400)

    filename = f"asl_{request.user.username}_{now().strftime('%Y%m%d%H%M%S')}.mp4"
    video_path = os.path.join(output_dir, filename)

    try:
        with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_url = f"{settings.MEDIA_URL}asl_videos/{filename}"
        video_size_kb = round(os.path.getsize(video_path) / 1024, 2)
        video_duration = len(frames) / fps

        # Save video history
        ASLVideoHistory.objects.create(
            user=request.user,
            input_text=text,
            frame_count=len(frames),
            video_size_kb=video_size_kb,
            video_duration_sec=video_duration,
            video_name=filename,
            video_url=video_url
        )

        # Audit log
        log_user_activity(
            request,
            action="Generate ASL Video",
            description=f"Generated video '{filename}' | Duration: {video_duration}s | Size: {video_size_kb}KB"
        )

        return JsonResponse({"video_url": video_url})

    except Exception as e:
        return JsonResponse({"error": "Video writing failed."}, status=500)


@login_required
def english_to_asl_view(request):
    # Log page visit
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited English to ASL translation page"
    )

    return render(request, "english_to_asl.html")


@login_required
def asl_video_history(request):
    # Log audit for page visit
    log_user_activity(
        request,
        action="Page Visit",
        description="Visited ASL Video History page"
    )

    user = request.user
    videos = ASLVideoHistory.objects.filter(user=user).order_by('-created_at')
    return render(request, 'asl_video_history.html', {'videos': videos})


@login_required
def admin_video_history_view(request):
    if not request.user.is_superuser:
        raise PermissionDenied("You do not have permission to view this page.")

    log_user_activity(request, action="Page Visit", description="Admin visited all users' ASL Video History")

    videos = ASLVideoHistory.objects.exclude(user=request.user).select_related('user').order_by('-created_at')

    return render(request, 'admin_video_history.html', {'videos': videos})

@login_required
def delete_prediction_view(request, prediction_id):
    try:
        if not request.user.is_superuser:
            return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

        prediction = get_object_or_404(ASLPrediction, id=prediction_id)
        prediction.delete()

        log_user_activity(
            request,
            action="Delete",
            description=f"Deleted Prediction ID: {prediction_id}"
        )

        return JsonResponse({'success': True, 'message': '✅ Prediction deleted successfully.'})

    except Exception as e:
        logger.error(f"❌ Error deleting prediction: {str(e)}")
        return JsonResponse({'success': False, 'message': 'Internal server error while deleting.'}, status=500)


@login_required
def delete_audit_log_view(request, log_id):
    try:
        if not request.user.is_superuser:
            return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

        log = get_object_or_404(AuditLog, id=log_id)
        log.delete()

        log_user_activity(
            request,
            action="Delete",
            description=f"Deleted AuditLog ID: {log_id}"
        )

        return JsonResponse({'success': True, 'message': '✅ Log entry deleted successfully.'})
    except Exception as e:
        print(f"❌ DELETE ERROR: {str(e)}")
        return JsonResponse({'success': False, 'message': 'Internal server error while deleting.'}, status=500)


@login_required
def delete_sentence_view(request, sentence_id):
    try:
        if not request.user.is_superuser:
            return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

        sentence = get_object_or_404(ASLSentenceGeneration, id=sentence_id)
        sentence.delete()

        log_user_activity(
            request,
            action="Delete",
            description=f"Deleted Sentence ID: {sentence_id}"
        )

        return JsonResponse({'success': True, 'message': '✅ Sentence entry deleted successfully.'})
    except Exception as e:
        print(f"❌ DELETE ERROR: {str(e)}")
        return JsonResponse({'success': False, 'message': 'Internal server error while deleting.'}, status=500)


@login_required
def delete_video_view(request, video_id):
    try:
        if not request.user.is_superuser:
            return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)

        video = get_object_or_404(ASLVideoHistory, id=video_id)
        video.delete()

        log_user_activity(
            request,
            action="Delete",
            description=f"Deleted Video History ID: {video_id}"
        )

        return JsonResponse({'success': True, 'message': '✅ Video record permanently deleted.'})
    except Exception as e:
        print(f"❌ DELETE ERROR: {str(e)}")
        return JsonResponse({'success': False, 'message': 'Internal server error while deleting.'}, status=500)