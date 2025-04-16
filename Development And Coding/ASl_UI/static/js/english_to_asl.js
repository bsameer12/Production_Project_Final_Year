document.getElementById('aslForm').addEventListener('submit', function(e) {
  e.preventDefault();

  const formData = new FormData();
  formData.append('text', document.getElementById('text').value);

  fetch(generateASLVideoURL, {
    method: 'POST',
    headers: { 'X-CSRFToken': csrfToken },
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    if (data.video_url) {
      const video = document.getElementById('aslPreview');
      const source = document.getElementById('aslSource');

      if (source) {
        source.src = data.video_url;
        video.load();
      } else {
        video.src = data.video_url;
      }

      document.getElementById('downloadLink').href = data.video_url;
      document.getElementById('downloadLink').download = 'asl_output.mp4';
      document.getElementById('previewSection').style.display = 'block';
    } else {
      alert("⚠️ No video returned. Please try again.");
    }
  })
  .catch(err => {
    console.error("⚠️ Fetch Error:", err);
    alert("⚠️ Error generating ASL video.");
  });
});
