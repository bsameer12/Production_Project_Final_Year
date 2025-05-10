const emailInput = document.getElementById('id_email');
    const emailError = document.getElementById('emailError');

    emailInput.addEventListener('input', () => {
      validateEmail();
    });

    function validateEmail() {
      const email = emailInput.value.trim();
      const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

      if (!email) {
        emailError.textContent = "Email is required.";
        return false;
      } else if (!regex.test(email)) {
        emailError.textContent = "Please enter a valid email address.";
        return false;
      } else {
        emailError.textContent = "";
        return true;
      }
    }