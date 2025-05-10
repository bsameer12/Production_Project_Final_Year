document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("registerForm");
  const inputs = form.querySelectorAll("input");
  const loader = document.getElementById("loader");
  const registerBtn = document.getElementById("registerBtn");

  // Fields
  const username = document.getElementById("id_username");
  const email = document.getElementById("id_email");
  const firstName = document.getElementById("id_first_name");
  const lastName = document.getElementById("id_last_name");
  const contact = document.getElementById("id_contact_number");
  const password1 = document.getElementById("id_password1");
  const password2 = document.getElementById("id_password2");

  // Patterns
  const namePattern = /^[A-Za-z\s]+$/;
  const contactPattern = /^[0-9]{7,15}$/;
  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  // Attach live validation
  inputs.forEach(input => {
    input.addEventListener("input", () => validateField(input));
  });

  form.addEventListener("submit", (e) => {
    let valid = true;
    inputs.forEach(input => {
      if (!validateField(input)) valid = false;
    });

    if (!valid) {
      e.preventDefault();
    } else {
      registerBtn.style.display = "none";
      loader.style.display = "block";
    }
  });

  function validateField(input) {
    const value = input.value.trim();
    const error = input.parentElement.querySelector(".error");
    const label = input.previousElementSibling?.innerText || "This field";

    if (!value) {
      return showError(input, error, `${label} is required.`);
    }

    switch (input.id) {
      case "id_username":
      case "id_first_name":
      case "id_last_name":
        if (!namePattern.test(value)) {
          return showError(input, error, `${label} must contain alphabets and spaces only.`);
        }
        break;

      case "id_email":
        if (!emailPattern.test(value)) {
          return showError(input, error, "Please enter a valid email address.");
        }
        break;

      case "id_contact_number":
        if (!contactPattern.test(value)) {
          return showError(input, error, "Contact must be 7–15 digits only.");
        }
        break;

      case "id_password1":
        const rules = [];
        if (value.length < 8) rules.push("at least 8 characters");
        if (!/[A-Z]/.test(value)) rules.push("one uppercase letter");
        if (!/[a-z]/.test(value)) rules.push("one lowercase letter");
        if (!/\d/.test(value)) rules.push("one number");
        if (!/[^\w\s]/.test(value)) rules.push("one special character");

        if (rules.length > 0) {
          return showError(input, error, `Password must contain ${rules.join(", ")}.`);
        }
        break;

      case "id_password2":
        if (value !== password1.value) {
          return showError(input, error, "Passwords do not match.");
        }
        break;
    }

    return clearError(input, error);
  }

  function showError(input, errorEl, message) {
    input.classList.remove("valid");
    input.classList.add("invalid");
    errorEl.textContent = message;
    errorEl.style.display = "block";
    return false;
  }

  function clearError(input, errorEl) {
    input.classList.remove("invalid");
    input.classList.add("valid");
    errorEl.textContent = "";
    errorEl.style.display = "none";
    return true;
  }

  // Auto dismiss alerts
  setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => el.remove());
  }, 9000);
});
