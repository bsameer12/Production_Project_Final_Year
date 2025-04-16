function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const screenWidth = window.innerWidth;

  if (screenWidth <= 767) {
    sidebar.classList.toggle('mobile-open');
  } else {
    sidebar.classList.toggle('collapsed');
  }
}

let currentPage = 1;
let rowsPerPage = 10;

function paginateTable() {
  const table = document.getElementById('predictionTable');
  const rows = Array.from(table.querySelectorAll('tbody tr')).filter(
    row => row.style.display !== 'none'
  );

  const totalPages = Math.ceil(rows.length / rowsPerPage);
  currentPage = Math.min(currentPage, totalPages || 1);

  const start = (currentPage - 1) * rowsPerPage;
  const end = start + rowsPerPage;

  let allRows = table.querySelectorAll('tbody tr');
  allRows.forEach(row => (row.style.display = 'none'));

  rows.slice(start, end).forEach(row => (row.style.display = ''));

  document.getElementById('currentPage').value = currentPage;
  updateRowInfo(rows.length, allRows.length);
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
  const visibleRows = Array.from(table.querySelectorAll('tbody tr')).filter(
    row => row.style.display !== 'none'
  );
  const totalPages = Math.ceil(visibleRows.length / rowsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    paginateTable();
  }
}

function filterTable() {
  const search = document.getElementById('searchInput').value.toLowerCase();
  const date = document.getElementById('dateFilter').value;
  const rows = document.querySelectorAll('#predictionTable tbody tr');

  rows.forEach(row => {
    const timestamp = row.children[0]?.textContent?.trim() || '';
    const rowDate = timestamp.split(' ')[0];
    const dateMatch = !date || rowDate === date;

    const textMatch = Array.from(row.children).some(td =>
      td.textContent.toLowerCase().includes(search)
    );

    row.style.display = dateMatch && textMatch ? '' : 'none';
  });

  currentPage = 1;
  paginateTable();
}

function updateRowInfo(visibleCount = 0, totalCount = 0) {
  const rowInfo = document.getElementById('rowInfo');
  rowInfo.textContent = `Showing ${visibleCount} of ${totalCount} records`;
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

  const csvContent = 'data:text/csv;charset=utf-8,' + csv.join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', 'prediction_history.csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Initial load
window.onload = () => {
  rowsPerPage = parseInt(document.getElementById('rowsPerPage').value);
  filterTable(); // also calls paginateTable
};

// Theme toggle
document.addEventListener('DOMContentLoaded', function () {
  const toggle = document.getElementById('themeSwitch');
  const label = document.getElementById('themeLabel');

  if (!toggle) return;

  const savedTheme = localStorage.getItem('theme') || 'light';
  document.body.setAttribute('data-theme', savedTheme);
  toggle.checked = savedTheme === 'dark';
  label.textContent = savedTheme === 'dark' ? 'Dark Mode' : 'Light Mode';

  toggle.addEventListener('change', () => {
    const theme = toggle.checked ? 'dark' : 'light';
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    label.textContent = theme === 'dark' ? 'Dark Mode' : 'Light Mode';
  });
});
