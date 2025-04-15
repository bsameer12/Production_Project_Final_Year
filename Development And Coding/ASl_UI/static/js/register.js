document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("registerForm");
    const inputs = form.querySelectorAll("input");
    const loader = document.getElementById("loader");
    const registerBtn = document.getElementById("registerBtn");

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
        const error = input.parentElement.querySelector(".error");
        if (input.value.trim() === "") {
            input.classList.remove("valid");
            input.classList.add("invalid");
            error.textContent = `${input.previousElementSibling.innerText} is required`;
            error.style.display = "block";
            return false;
        } else {
            input.classList.remove("invalid");
            input.classList.add("valid");
            error.textContent = "";
            error.style.display = "none";
            return true;
        }
    }
});
setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => el.remove());
  }, 9000);