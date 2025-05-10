document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loginForm");
  const usernameInput = document.getElementById("id_username");
  const passwordInput = document.getElementById("id_password");
  const loader = document.getElementById("loader");
  const loginBtn = document.getElementById("loginBtn");

  const inputs = form.querySelectorAll("input");

  // 🧠 Attach live validation to inputs
  inputs.forEach(input => {
    input.addEventListener("input", () => validateField(input));
  });

  // ✅ Form submit
  form.addEventListener("submit", (e) => {
    let isValid = true;

    inputs.forEach(input => {
      if (!validateField(input)) isValid = false;
    });

    if (!isValid) {
      e.preventDefault();
    } else {
      loginBtn.style.display = "none";
      loader.style.display = "block";
    }
  });

  // 🔎 Field validation function
  function validateField(input) {
    const value = input.value.trim();
    const error = input.parentElement.querySelector(".error");

    if (input.id === "id_username") {
      if (!value) {
        return showError(input, error, "Username is required.");
      }
    }

    if (input.id === "id_password") {
      const errors = [];

      if (value.length < 8) errors.push("at least 8 characters");
      if (!/[A-Z]/.test(value)) errors.push("one uppercase letter");
      if (!/[a-z]/.test(value)) errors.push("one lowercase letter");
      if (!/\d/.test(value)) errors.push("one number");
      if (!/[^\w\s]/.test(value)) errors.push("one special character");

      if (errors.length > 0) {
        return showError(input, error, `Password must contain ${errors.join(', ')}.`);
      }
    }

    return clearError(input, error);
  }

  // ⚠️ Show error
  function showError(input, errorElement, message) {
    input.classList.remove("valid");
    input.classList.add("invalid");
    errorElement.textContent = message;
    errorElement.style.display = "block";
    return false;
  }

  // ✅ Clear error
  function clearError(input, errorElement) {
    input.classList.remove("invalid");
    input.classList.add("valid");
    errorElement.textContent = "";
    errorElement.style.display = "none";
    return true;
  }

  // ⏳ Remove alert messages after 9s
  setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => el.remove());
  }, 9000);
});

