let stream = null;
let predicting = false;
let ttsEnabled = true;

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('webcam').srcObject = stream;
    updateCameraStatus(true);
  } catch (err) {
    alert("Could not access webcam.");
    console.error(err);
    updateCameraStatus(false);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    document.getElementById('webcam').srcObject = null;
    updateCameraStatus(false);
  }
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

function startPrediction() {
  predicting = true;
  document.getElementById("prediction").textContent = "Predicting...";
  // Add your model inference call here
}

function stopPrediction() {
  predicting = false;
  document.getElementById("prediction").textContent = "-";
  document.getElementById("confidence").textContent = "0.00";
}

function clearPrediction() {
  document.getElementById("prediction").textContent = "-";
  document.getElementById("confidence").textContent = "0.00";
}

function toggleTTS() {
  ttsEnabled = document.getElementById("ttsToggle").checked;
}

document.addEventListener("DOMContentLoaded", () => {
  updateCameraStatus(false); // Initialize on load
});
