const apiBase = "http://localhost:8000/api/v1";

const healthButton = document.getElementById("health-check-button");
const healthOutput = document.getElementById("health-output");
const generateNoteForm = document.getElementById("generate-note-form");
const noteOutput = document.getElementById("note-output");

async function checkHealth() {
  healthOutput.textContent = "Checking backend...";
  try {
    const response = await fetch(`${apiBase}/health`);
    const json = await response.json();
    healthOutput.textContent = JSON.stringify(json, null, 2);
  } catch (error) {
    healthOutput.textContent = `Unable to reach backend: ${error}`;
  }
}

async function generateNote(event) {
  event.preventDefault();
  noteOutput.textContent = "Generating note...";

  const formData = new FormData(generateNoteForm);
  const payload = {
    patient_id: Number(formData.get("patient_id")),
    video_id: formData.get("video_id"),
    video_path: formData.get("video_path"),
    camera_id: formData.get("camera_id") || null,
    additional_context: formData.get("additional_context") || null,
  };

  try {
    const response = await fetch(`${apiBase}/observation-notes/generate-from-detection`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const json = await response.json();
    noteOutput.textContent = JSON.stringify(json, null, 2);
  } catch (error) {
    noteOutput.textContent = `Request failed: ${error}`;
  }
}

healthButton.addEventListener("click", checkHealth);
generateNoteForm.addEventListener("submit", generateNote);
