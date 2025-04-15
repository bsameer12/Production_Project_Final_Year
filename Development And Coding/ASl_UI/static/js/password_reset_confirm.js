const passwordInput = document.getElementById('id_new_password1');
const confirmInput = document.getElementById('id_new_password2');
const errorList = document.getElementById('passwordErrorList');
const strengthBar = document.getElementById('strengthBar');

passwordInput.addEventListener('input', updateValidation);
confirmInput.addEventListener('input', updateValidation);

function updateValidation() {
  const password = passwordInput.value;
  const confirm = confirmInput.value;
  const errors = [];

  const length = password.length >= 8;
  const uppercase = /[A-Z]/.test(password);
  const lowercase = /[a-z]/.test(password);
  const number = /[0-9]/.test(password);
  const special = /[^A-Za-z0-9]/.test(password);
  const match = password === confirm;

  if (!length) errors.push("✔️ At least 8 characters");
  if (!uppercase) errors.push("✔️ At least one uppercase letter");
  if (!lowercase) errors.push("✔️ At least one lowercase letter");
  if (!number) errors.push("✔️ At least one number");
  if (!special) errors.push("✔️ At least one special character");
  if (!match && confirm.length > 0) errors.push("❌ Passwords do not match");

  errorList.innerHTML = errors.length ? errors.join('<br>') : "";

  const score = [length, uppercase, lowercase, number, special].filter(Boolean).length;
  strengthBar.style.width = `${(score / 5) * 100}%`;
  strengthBar.style.backgroundColor = score >= 4 ? "#22c55e" : score === 3 ? "#facc15" : "#ef4444";
}

function validatePassword() {
  updateValidation();
  return errorList.innerHTML === "";
}
