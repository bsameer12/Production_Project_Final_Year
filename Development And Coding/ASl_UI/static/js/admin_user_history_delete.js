
  function confirmDeleteLog(id) {
  const confirmed = confirm("Are you sure you want to delete this log entry?");
  if (confirmed) {
    fetch(`/delete-audit-log/${id}/`, {
      method: 'POST',
      headers: {
        'X-CSRFToken': getCookie('csrftoken'),
        'Content-Type': 'application/json',
      }
    })
    .then(res => {
      if (!res.ok) throw new Error('Network response not ok');
      return res.json();
    })
    .then(data => {
      showDeleteMessage(data.success, data.message);
      if (data.success) setTimeout(() => location.reload(), 900);
    })
    .catch(() => {
      showDeleteMessage(false, 'An error occurred while trying to delete.');
    });
  }
}
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    document.cookie.split(';').forEach(cookie => {
      const trimmed = cookie.trim();
      if (trimmed.startsWith(name + '=')) {
        cookieValue = decodeURIComponent(trimmed.substring(name.length + 1));
      }
    });
  }
  return cookieValue;
}

function showDeleteMessage(success, message) {
  const msg = document.getElementById('deleteMessage');
  msg.textContent = message;
  msg.className = 'delete-message ' + (success ? 'success' : 'error');
  msg.style.display = 'block';
  setTimeout(() => msg.style.display = 'none', 9000);
}
