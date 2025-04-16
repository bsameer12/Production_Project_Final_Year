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
    let currentPage = 1;
    let rowsPerPage = 10;

    function paginateTable() {
      const table = document.getElementById('predictionTable');
      const rows = table.querySelectorAll('tbody tr');
      const start = (currentPage - 1) * rowsPerPage;
      const end = start + rowsPerPage;

      rows.forEach((row, index) => {
        row.style.display = (index >= start && index < end) ? '' : 'none';
      });

      document.getElementById('currentPage').value = currentPage;
    }

    function changeRowsPerPage() {
      rowsPerPage = parseInt(document.getElementById('rowsPerPage').value);
      currentPage = 1;
      paginateTable();
    }

    function goToPage() {
      currentPage = parseInt(document.getElementById('currentPage').value);
      paginateTable();
    }

    function prevPage() {
      if (currentPage > 1) {
        currentPage--;
        paginateTable();
      }
    }

    function nextPage() {
      const table = document.getElementById('predictionTable');
      const rows = table.querySelectorAll('tbody tr');
      const totalPages = Math.ceil(rows.length / rowsPerPage);
      if (currentPage < totalPages) {
        currentPage++;
        paginateTable();
      }
    }

    function filterTable() {
  const search = document.getElementById('searchInput').value.toLowerCase();
  const date = document.getElementById('dateFilter').value;
  const table = document.getElementById('predictionTable');
  const rows = table.querySelectorAll('tbody tr');

  rows.forEach(row => {
    const timestampCell = row.children[0].textContent.trim(); // e.g., "2025-04-16 10:32:45"
    const rowDate = timestampCell.split(' ')[0]; // Extract "2025-04-16"

    const dateMatch = !date || rowDate === date;

    const textMatch = [...row.children].some(td =>
      td.textContent.toLowerCase().includes(search)
    );

    row.style.display = dateMatch && textMatch ? '' : 'none';
  });
}


    function exportTableToCSV() {
      const table = document.getElementById('predictionTable');
      const rows = table.querySelectorAll('tr');
      let csv = [];

      rows.forEach(row => {
        const cols = row.querySelectorAll('th, td');
        const rowData = [...cols].map(col => `"${col.textContent.trim()}"`);
        csv.push(rowData.join(','));
      });

      const csvContent = "data:text/csv;charset=utf-8," + csv.join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "prediction_history.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    // Initial setup
    window.onload = () => {
      rowsPerPage = parseInt(document.getElementById('rowsPerPage').value);
      paginateTable();
    };

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

