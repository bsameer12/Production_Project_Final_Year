let stream = null;
let predicting = false;
let landmarkQueue = [];
let ttsEnabled = true;
let hands = null;
let camera = null;
let cooldown = false;
let handPreviouslyDetected = false;
let lastPrediction = null;
let lastTTSLabel = null;

let predictionChart, scatterChart;
let predictionSequence = []; // store top-1 predictions

function appendToHistory(label, confidence) {
  predictionSequence.push(label);

  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleString()} → ${label} (${confidence})`;
  document.getElementById("predictionHistory").prepend(item);

  // Live output preview
  document.getElementById("sentence-output").value = predictionSequence.join('');
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.toggle(window.innerWidth <= 767 ? 'mobile-open' : 'collapsed');
}

function updateCameraStatus(active) {
  document.getElementById("camera-dot").className = "dot " + (active ? "green" : "red");
  document.getElementById("camera-status-text").textContent = active ? "Camera started" : "Camera stopped";
}

function playBeep() {
  new Audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg").play().catch(() => {});
}

function getCSRFToken() {
  const match = document.cookie.match(/csrftoken=([^;]+)/);
  return match ? match[1] : '';
}

function arraysEqual(a, b) {
  return a && b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 0.0001);
}

function startCooldown() {
  cooldown = true;
  setTimeout(() => (cooldown = false), 5000);
}

function updateLandmarkList(landmarks) {
  document.getElementById("landmarkPoints").innerHTML = landmarks.map(
    (pt, i) => `<div>Point ${i}: x=${pt.x.toFixed(3)}, y=${pt.y.toFixed(3)}, z=${pt.z.toFixed(3)}</div>`
  ).join("");
  updateScatterChart(landmarks);
}

function clearPrediction() {
  document.getElementById("prediction").textContent = "-";
  document.getElementById("confidence").textContent = "0.00";
  document.getElementById("top2").textContent = "-";
  document.getElementById("top3").textContent = "-";
  document.getElementById("predictionHistory").innerHTML = "";
  document.getElementById("landmarkPoints").innerHTML = "";
  document.getElementById("sentence-output").value = "";
  landmarkQueue = [];
  predictionSequence = [];
  lastPrediction = null;
  lastTTSLabel = null;
}

function startPrediction() {
  predicting = true;
  startCooldown();
  document.getElementById("feedbackBox").textContent = "Prediction Running...";
  document.getElementById("sentence-output").value = predictionSequence.join('');
}

function stopPrediction() {
 console.log("🛑 stopPrediction() triggered");
  predicting = false;
  document.getElementById("feedbackBox").textContent = "Prediction Paused.";

  if (predictionSequence.length > 0) {
    console.log("🟢 Letters collected:", predictionSequence);

    fetch("/generate_sentence/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCSRFToken()
      },
      body: JSON.stringify({ predictions: predictionSequence })
    })
    .then(r => {
      if (!r.ok) throw new Error(`Server returned ${r.status}`);
      return r.json();
    })
    .then(data => {
      console.log("✅ ChatGPT Response:", data);
      const sentenceBox = document.getElementById("sentence-output");

      if (data.sentence) {
        sentenceBox.value = data.sentence;
        if (ttsEnabled) {
          window.speechSynthesis.speak(new SpeechSynthesisUtterance(data.sentence));
        }
      } else if (data.error) {
        sentenceBox.value = `⚠️ ChatGPT Error: ${data.error}`;
      }
    })
    .catch(error => {
      console.error("❌ Error sending to backend:", error);
      document.getElementById("sentence-output").value = `⚠️ Error: ${error.message}`;
    });
  } else {
    console.warn("⚠️ No predictions to send to ChatGPT.");
    document.getElementById("sentence-output").value = "⚠️ No predictions to send.";
  }
}


function toggleTTS() {
  ttsEnabled = document.getElementById("ttsToggle").checked;
}

async function startCamera() {
  const canvas = document.getElementById("webcam-canvas");
  const ctx = canvas.getContext("2d");
  const video = document.createElement("video");
  video.autoplay = video.playsInline = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    updateCameraStatus(true);

    hands = new Hands({
      locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
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
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      const hand = results.multiHandLandmarks?.[0];
      if (!hand) {
        if (handPreviouslyDetected) {
          handPreviouslyDetected = false;
          document.getElementById("feedbackBox").textContent = "Hand not detected.";
        }
        if (!predicting) document.getElementById("feedbackBox").textContent = "Prediction Off";
        ctx.restore();
        return;
      }

      if (!handPreviouslyDetected) {
        playBeep();
        document.getElementById("feedbackBox").textContent = "Hand detected.";
        startCooldown();
        handPreviouslyDetected = true;
      }

      drawConnectors(ctx, hand, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
      drawLandmarks(ctx, hand, { color: '#FF0000', lineWidth: 1 });
      ctx.restore();
      updateLandmarkList(hand);

      if (predicting && !cooldown) {
        const wrist = hand[0];
        const normed = hand.map(p => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]).flat();
        if (!arraysEqual(normed, landmarkQueue[landmarkQueue.length - 1])) {
          landmarkQueue.push(normed);
        }

        if (landmarkQueue.length === 10) {
          const sequence = [...landmarkQueue];
          landmarkQueue = [];
          startCooldown();

          fetch("/predict_landmarks/", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-CSRFToken": getCSRFToken()
            },
            body: JSON.stringify({ sequence })
          })
          .then(r => r.json())
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
            document.getElementById("feedbackBox").textContent = `Prediction: ${data.label} (${data.confidence})`;

            appendToHistory(data.label, data.confidence);
            updatePredictionChart(data);

            if (ttsEnabled && data.confidence > 0.75 && data.label !== lastTTSLabel) {
              window.speechSynthesis.speak(new SpeechSynthesisUtterance(data.label));
              lastTTSLabel = data.label;
            }
          });
        }
      }
    });

    camera = new Camera(video, {
      onFrame: async () => await hands.send({ image: video }),
      width: canvas.width,
      height: canvas.height
    });
    camera.start();

  } catch (e) {
    alert("Webcam access denied.");
    updateCameraStatus(false);
  }
}

function initPredictionChart() {
  const ctx = document.getElementById("predictionBarChart").getContext("2d");
  predictionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Top 1', 'Top 2', 'Top 3'],
      datasets: [{
        label: 'Confidence',
        backgroundColor: ['#007bff', '#28a745', '#ffc107'],
        data: [0, 0, 0]
      }]
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  });
}

function updatePredictionChart(data) {
  predictionChart.data.labels = [data.label, data.top2.label, data.top3.label];
  predictionChart.data.datasets[0].data = [data.confidence, data.top2.confidence, data.top3.confidence];
  predictionChart.update();
}

function initScatterChart() {
  const ctx = document.getElementById('landmarkScatterChart').getContext('2d');
  scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: 'Landmark (x, y)', data: [], backgroundColor: 'red' }] },
    options: {
      responsive: true,
      scales: {
        x: { type: 'linear', min: 0, max: 1 },
        y: { type: 'linear', min: 0, max: 1 }
      }
    }
  });
}

function updateScatterChart(landmarks) {
  scatterChart.data.datasets[0].data = landmarks.map(pt => ({ x: pt.x, y: pt.y }));
  scatterChart.update();
}

function stopCamera() {
  if (stream) stream.getTracks().forEach(track => track.stop());
  if (camera) camera.stop();
  updateCameraStatus(false);
  document.getElementById("feedbackBox").textContent = "Camera stopped.";
}

document.addEventListener("DOMContentLoaded", () => {
  updateCameraStatus(false);
  initPredictionChart();
  initScatterChart();
});
