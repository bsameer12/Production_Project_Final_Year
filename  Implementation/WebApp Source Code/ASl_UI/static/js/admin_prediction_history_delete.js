
    function confirmDelete(id) {
  const confirmed = confirm("⚠️ Are you sure you want to delete this entry?");
  if (confirmed) {
    fetch(`/delete-prediction/${id}/`, {
      method: 'POST',
      headers: {
        'X-CSRFToken': getCookie('csrftoken'),
        'Content-Type': 'application/json',
      }
    })
    .then(response => response.json())
    .then(data => {
      showDeleteMessage(data.success, data.message || "Deleted successfully.");

      if (data.success) {
        setTimeout(() => {
          location.reload();
        }, 900); // slight delay before reload for smoother fade
      }
    })
    .catch(() => {
      showDeleteMessage(false, "An error occurred while trying to delete.");
    });
  }
}

function showDeleteMessage(success, message) {
  const msgEl = document.getElementById("deleteMessage");
  msgEl.textContent = message;
  msgEl.className = "delete-message " + (success ? "success" : "error");
  msgEl.style.display = "block";

  setTimeout(() => {
    msgEl.style.display = "none";
  }, 9000);
}

function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
      cookie = cookie.trim();
      if (cookie.startsWith(name + "=")) {
        cookieValue = decodeURIComponent(cookie.slice(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}
