let stream = null;
let predicting = false;
let landmarkQueue = [];
let ttsEnabled = true;
let hands = null;
let camera = null;
let handPreviouslyDetected = false; // Used to trigger sound only once

// Toggle sidebar visibility
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
}

// Start camera and MediaPipe Hands
async function startCamera() {
  const canvas = document.getElementById("webcam-canvas");
  const ctx = canvas.getContext("2d");

  const video = document.createElement("video");
  video.width = canvas.width;
  video.height = canvas.height;
  video.autoplay = true;
  video.playsInline = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    updateCameraStatus(true);

    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults((results) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      const landmarksDetected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

      // 🔔 Play sound once when hand is detected
      if (landmarksDetected && !handPreviouslyDetected) {
        playBeep();
        handPreviouslyDetected = true;
        document.getElementById("feedbackBox").textContent = "Hand detected.";
      }

      if (!landmarksDetected) {
        handPreviouslyDetected = false;
        document.getElementById("feedbackBox").textContent = "Hand not detected.";
        return;
      }

      if (predicting) {
        const landmarks = results.multiHandLandmarks[0];
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
        drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 1 });

        const wrist = landmarks[0];
        const normed = landmarks.map(pt => [pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z]);
        const flat = normed.flat();

        landmarkQueue.push(flat);
        if (landmarkQueue.length === 10) {
          const sequence = [...landmarkQueue];
          landmarkQueue = [];

          fetch("/predict_landmarks/", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-CSRFToken": getCSRFToken()
            },
            body: JSON.stringify({ sequence: sequence })
          })
            .then(res => res.json())
            .then(data => {
              if (data.error) {
                document.getElementById("feedbackBox").textContent = "Prediction failed: " + data.error;
                return;
              }

              document.getElementById("prediction").textContent = data.label;
              document.getElementById("confidence").textContent = data.confidence;
              document.getElementById("feedbackBox").textContent = `Prediction: ${data.label} (${data.confidence})`;

              if (ttsEnabled && data.confidence > 0.7) {
                const utter = new SpeechSynthesisUtterance(data.label);
                window.speechSynthesis.speak(utter);
              }
            })
            .catch(err => {
              document.getElementById("feedbackBox").textContent = "Prediction error: " + err.message;
            });
        }
      }
    });

    camera = new Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video });
      },
      width: canvas.width,
      height: canvas.height
    });

    camera.start();
  } catch (error) {
    alert("Webcam access denied.");
    updateCameraStatus(false);
    console.error(error);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  if (camera) camera.stop();
  updateCameraStatus(false);
  document.getElementById("feedbackBox").textContent = "Camera stopped.";
}

function startPrediction() {
  predicting = true;
  document.getElementById("feedbackBox").textContent = "Prediction Running...";
}

function stopPrediction() {
  predicting = false;
  document.getElementById("feedbackBox").textContent = "Prediction Paused.";
}

function clearPrediction() {
  document.getElementById("prediction").textContent = "-";
  document.getElementById("confidence").textContent = "0.00";
  document.getElementById("feedbackBox").textContent = "Prediction Cleared";
  landmarkQueue = [];
}

function toggleTTS() {
  ttsEnabled = document.getElementById("ttsToggle").checked;
}

function updateCameraStatus(active) {
  const dot = document.getElementById("camera-dot");
  const text = document.getElementById("camera-status-text");
  if (active) {
    dot.classList.remove("red");
    dot.classList.add("green");
    text.textContent = "Camera started";
  } else {
    dot.classList.remove("green");
    dot.classList.add("red");
    text.textContent = "Camera stopped";
  }
}

function getCSRFToken() {
  const cookieValue = document.cookie.match('(^|;)\\s*csrftoken\\s*=\\s*([^;]+)');
  return cookieValue ? cookieValue.pop() : '';
}

// 🔔 Sound when hand is detected
function playBeep() {
  const audio = new Audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg");
  audio.play().catch(e => console.warn("Autoplay policy may block audio: ", e));
}

document.addEventListener("DOMContentLoaded", () => {
  updateCameraStatus(false);
});
