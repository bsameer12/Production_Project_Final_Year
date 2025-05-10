
   function confirmDelete(username, deleteUrl) {
  if (confirm(`Are you sure you want to delete user: ${username}?`)) {
    fetch(deleteUrl, {
      method: 'POST',
      headers: {
        'X-CSRFToken': getCookie('csrftoken'),
        'Content-Type': 'application/json',
      }
    })
    .then(response => response.json())
    .then(data => {
      showDeleteMessage(data.success, data.message);
      if (data.success) {
        setTimeout(() => location.reload(), 1000);
      }
    })
    .catch(() => {
      showDeleteMessage(false, 'An error occurred while deleting the user.');
    });
  }
}


    function showDeleteMessage(success, message) {
      const box = document.getElementById('deleteMessage');
      box.className = `delete-message ${success ? 'success' : 'error'}`;
      box.textContent = message;
      box.style.display = 'block';
      setTimeout(() => { box.style.display = 'none'; }, 9000);
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