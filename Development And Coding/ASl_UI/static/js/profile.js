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

document.addEventListener("DOMContentLoaded", function () {
    const toggle = document.getElementById("themeSwitch");
    const label = document.getElementById("themeLabel");

    const savedTheme = localStorage.getItem("theme") || "light";
    document.body.setAttribute("data-theme", savedTheme);
    toggle.checked = savedTheme === "dark";
    label.textContent = savedTheme === "dark" ? "Dark Mode" : "Light Mode";

    toggle.addEventListener("change", () => {
      const theme = toggle.checked ? "dark" : "light";
      document.body.setAttribute("data-theme", theme);
      localStorage.setItem("theme", theme);
      label.textContent = theme === "dark" ? "Dark Mode" : "Light Mode";
    });
  });