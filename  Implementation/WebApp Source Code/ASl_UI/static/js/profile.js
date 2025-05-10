// Theme Toggle

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("themeSwitch");
  const label = document.getElementById("themeLabel");

  // Apply saved theme
  const savedTheme = localStorage.getItem("theme") || "light";
  document.body.setAttribute("data-theme", savedTheme);
  if (toggle) toggle.checked = savedTheme === "dark";
  if (label) label.textContent = savedTheme === "dark" ? "Dark Mode" : "Light Mode";

  // Handle theme switch
  toggle?.addEventListener("change", () => {
    const theme = toggle.checked ? "dark" : "light";
    document.body.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    label.textContent = theme === "dark" ? "Dark Mode" : "Light Mode";
  });

  // Auto-dismiss alerts
  setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => el.remove());
  }, 9000);

  // Live validation fields
  const firstName = document.querySelector('input[name="first_name"]');
  const lastName = document.querySelector('input[name="last_name"]');
  const contact = document.querySelector('input[name="contact_number"]');
  const password = document.getElementById('id_new_password1');

  const rules = {
    length: document.getElementById('rule-length'),
    upper: document.getElementById('rule-uppercase'),
    lower: document.getElementById('rule-lowercase'),
    number: document.getElementById('rule-number'),
    special: document.getElementById('rule-special'),
  };
  const strengthBox = document.getElementById('password-strength');

  function showError(input, message) {
    let error = input.parentNode.querySelector('.input-error');
    if (!error) {
      error = document.createElement('div');
      error.className = 'input-error';
      input.insertAdjacentElement('afterend', error);
    }
    error.textContent = message;
    error.style.display = 'block';
  }

  function clearError(input) {
    const error = input.parentNode.querySelector('.input-error');
    if (error) {
      error.textContent = '';
      error.style.display = 'none';
    }
  }

  // Name Validation: Only letters and spaces
  [firstName, lastName].forEach(field => {
    field?.addEventListener('input', () => {
      const value = field.value.trim();
      const pattern = /^[A-Za-z\s]+$/;

      if (!value) {
        showError(field, 'This field is required.');
      } else if (!pattern.test(value)) {
        showError(field, 'Only alphabets and spaces allowed.');
      } else {
        clearError(field);
      }
    });
  });

  // Contact number (optional, numeric check if provided)
  contact?.addEventListener('input', () => {
    const val = contact.value.trim();
    if (val && !/^\d+$/.test(val)) {
      showError(contact, 'Contact number must be digits only.');
    } else {
      clearError(contact);
    }
  });

  // Password Strength Validation
  password?.addEventListener('input', () => {
    const val = password.value;
    let score = 0;

    const checks = [
      { test: val.length >= 8, rule: rules.length },
      { test: /[A-Z]/.test(val), rule: rules.upper },
      { test: /[a-z]/.test(val), rule: rules.lower },
      { test: /\d/.test(val), rule: rules.number },
      { test: /[^A-Za-z0-9]/.test(val), rule: rules.special }
    ];

    checks.forEach(({ test, rule }) => {
      if (!rule) return;
      if (test) {
        rule.classList.add('valid');
        score++;
      } else {
        rule.classList.remove('valid');
      }
    });

    // Update strength bar
    if (strengthBox) {
      strengthBox.className = 'password-strength';
      if (score <= 2) {
        strengthBox.classList.add('weak');
        strengthBox.textContent = 'Weak';
      } else if (score <= 4) {
        strengthBox.classList.add('medium');
        strengthBox.textContent = 'Medium';
      } else {
        strengthBox.classList.add('strong');
        strengthBox.textContent = 'Strong';
      }
    }
  });
});

// Sidebar toggle for responsive view
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const isMobile = window.innerWidth <= 767;
  sidebar.classList.toggle(isMobile ? 'mobile-open' : 'collapsed');
}
