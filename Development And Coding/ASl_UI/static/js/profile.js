    function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const screenWidth = window.innerWidth;

  if (screenWidth <= 767) {
    // Mobile: show/hide full sidebar
    sidebar.classList.toggle('mobile-open');
  } else {
    // Desktop & Tablet: expand/collapse sidebar
    sidebar.classList.toggle('collapsed');
  }
}