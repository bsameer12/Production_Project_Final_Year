
  function confirmDeleteSentence(id) {
  const confirmed = confirm("Are you sure you want to delete this sentence entry?");
  if (confirmed) {
    fetch(`/delete-sentence/${id}/`, {
      method: 'POST',
      headers: {
        'X-CSRFToken': getCookie('csrftoken'),
        'Content-Type': 'application/json',
      }
    })
    .then(response => {
      if (!response.ok) throw new Error('Delete failed');
      return response.json();
    })
    .then(data => {
      showDeleteMessage(data.success, data.message);
      if (data.success) setTimeout(() => location.reload(), 1000);
    })
    .catch(() => {
      showDeleteMessage(false, 'An error occurred while deleting.');
    });
  }
}

function showDeleteMessage(success, message) {
  const msgEl = document.getElementById('deleteMessage');
  msgEl.textContent = message;
  msgEl.className = 'delete-message ' + (success ? 'success' : 'error');
  msgEl.style.display = 'block';
  setTimeout(() => {
    msgEl.style.display = 'none';
  }, 9000);
}

function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    document.cookie.split(';').forEach(cookie => {
      const cookieTrimmed = cookie.trim();
      if (cookieTrimmed.startsWith(name + '=')) {
        cookieValue = decodeURIComponent(cookieTrimmed.slice(name.length + 1));
      }
    });
  }
  return cookieValue;
}
