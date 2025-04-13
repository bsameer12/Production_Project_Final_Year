let stream = null;
let predicting = false;
let landmarkQueue = [];
let ttsEnabled = true;
let hands = null;
let camera = null;
let handPreviouslyDetected = false;
let lastPrediction = null;
let lastTTSLabel = null;

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
}

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
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    hands.onResults(results => {
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Mirror canvas
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);

      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      const detected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

      if (!detected) {
        if (handPreviouslyDetected) {
          handPreviouslyDetected = false;
          document.getElementById("feedbackBox").textContent = "Hand not detected.";
          clearPrediction();
        }

        if (!predicting) {
          document.getElementById("feedbackBox").textContent = "Prediction Off";
        }

        ctx.restore();
        return;
      }

      if (!handPreviouslyDetected) {
        playBeep();
        handPreviouslyDetected = true;
        document.getElementById("feedbackBox").textContent = "Hand detected.";
      }

      const landmarks = results.multiHandLandmarks[0];

      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
      drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 1 });

      ctx.restore();

      updateLandmarkList(landmarks); // Show landmark values

      if (predicting) {
        const wrist = landmarks[0];
        const normed = landmarks.map(pt => [pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z]);
        const flat = normed.flat();

        if (landmarkQueue.length === 0 || !arraysEqual(flat, landmarkQueue[landmarkQueue.length - 1])) {
          landmarkQueue.push(flat);
        }

        if (landmarkQueue.length === 10) {
          const sequence = [...landmarkQueue];
          landmarkQueue = [];

          fetch("/predict_landmarks/", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-CSRFToken": getCSRFToken()
            },
            body: JSON.stringify({ sequence })
          })
            .then(res => res.json())
            .then(data => {
              if (data.error) {
                document.getElementById("feedbackBox").textContent = data.error;
                return;
              }

              if (data.label === lastPrediction) {
                document.getElementById("feedbackBox").textContent = "Prediction Paused: Same Gesture";
                return;
              }

              lastPrediction = data.label;

              document.getElementById("prediction").textContent = data.label;
              document.getElementById("confidence").textContent = data.confidence;
              document.getElementById("top2").textContent = `${data.top2.label} (${data.top2.confidence})`;
              document.getElementById("top3").textContent = `${data.top3.label} (${data.top3.confidence})`;

              document.getElementById("feedbackBox").textContent =
                `Prediction: ${data.label} (${data.confidence})`;

              appendToHistory(data.label, data.confidence);

              if (ttsEnabled && data.confidence > 0.75 && data.label !== lastTTSLabel) {
                const utter = new SpeechSynthesisUtterance(data.label);
                window.speechSynthesis.speak(utter);
                lastTTSLabel = data.label;
              }
            })
            .catch(err => {
              document.getElementById("feedbackBox").textContent = "Prediction error: " + err.message;
            });
        }
      } else {
        document.getElementById("feedbackBox").textContent = "Prediction Off";
      }
    });

    camera = new Camera(video, {
      onFrame: async () => await hands.send({ image: video }),
      width: canvas.width,
      height: canvas.height
    });

    camera.start();
  } catch (error) {
    alert("Webcam access denied.");
    console.error(error);
    updateCameraStatus(false);
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach(track => track.stop());
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
  document.getElementById("top2").textContent = "-";
  document.getElementById("top3").textContent = "-";
  landmarkQueue = [];
  lastPrediction = null;
  document.getElementById("predictionHistory").innerHTML = "";
  document.getElementById("landmarkPoints").innerHTML = "";
}

function toggleTTS() {
  ttsEnabled = document.getElementById("ttsToggle").checked;
}

function updateCameraStatus(active) {
  const dot = document.getElementById("camera-dot");
  const text = document.getElementById("camera-status-text");
  dot.className = "dot " + (active ? "green" : "red");
  text.textContent = active ? "Camera started" : "Camera stopped";
}

function getCSRFToken() {
  const cookie = document.cookie.match('(^|;)\\s*csrftoken\\s*=\\s*([^;]+)');
  return cookie ? cookie.pop() : '';
}

function playBeep() {
  const beep = new Audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg");
  beep.play().catch(e => console.warn("Autoplay policy prevented sound"));
}

function arraysEqual(arr1, arr2) {
  if (!arr1 || !arr2 || arr1.length !== arr2.length) return false;
  for (let i = 0; i < arr1.length; i++) {
    if (Math.abs(arr1[i] - arr2[i]) > 0.0001) return false;
  }
  return true;
}

// 👇 Show live landmark points in a scrollable list
function updateLandmarkList(landmarks) {
  const container = document.getElementById("landmarkPoints");
  container.innerHTML = landmarks.map((pt, i) =>
    `<div>Point ${i}: x=${pt.x.toFixed(3)}, y=${pt.y.toFixed(3)}, z=${pt.z.toFixed(3)}</div>`
  ).join("");
}

// 👇 Add to prediction history below
function appendToHistory(label, confidence) {
  const now = new Date();
  const timestamp = now.toLocaleString();
  const item = document.createElement("li");
  item.textContent = `${timestamp} → ${label} (${confidence})`;
  document.getElementById("predictionHistory").prepend(item);
}

document.addEventListener("DOMContentLoaded", () => {
  updateCameraStatus(false);
});
