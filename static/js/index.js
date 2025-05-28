import "htmx.org";

export function enableSubmitOnValidInput({
  inputSelector = "#youtube-url",
  submitSelector = "#submit-btn",
} = {}) {
  const input = document.querySelector(inputSelector);
  const submitBtn = document.querySelector(submitSelector);

  if (!input || !submitBtn) return;

  const toggleSubmit = () => {
    submitBtn.disabled = !input.checkValidity();
  };

  input.addEventListener("input", toggleSubmit);
  window.addEventListener("DOMContentLoaded", toggleSubmit);
}

// Automatically enable on default form if present
enableSubmitOnValidInput();