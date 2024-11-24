// Select form elements
const form = document.querySelector("form");
const adviceDisplay = document.querySelector(".border p");

// Event listener for form submission
form.addEventListener("submit", async (event) => {
  event.preventDefault(); // Prevent the default form submission behavior

  // Get values from input fields
  const temperature = form.querySelector("input[placeholder='Enter Temperature (in °C)']").value;
  const rainfall = form.querySelector("input[placeholder='Enter Rainfall (in mm)']").value;
  const windSpeed = form.querySelector("input[placeholder='Enter Wind Speed (in km/h)']").value;
  const humidity = form.querySelector("input[placeholder='Enter Humidity (in %)']").value;

  // Validate inputs
  if (!temperature || !rainfall || !windSpeed || !humidity) {
    adviceDisplay.textContent = "Please fill in all fields.";
    adviceDisplay.style.color = "red";
    return;
  }

  // Show loading message
  adviceDisplay.textContent = "Fetching advice...";
  adviceDisplay.style.color = "gray";

  try {
    // Simulated API call (replace with your API URL)
    const response = await fetch("https://7e8c-34-83-105-230.ngrok-free.app/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        temperature: parseFloat(temperature),
        rainfall: parseFloat(rainfall),
        windSpeed: parseFloat(windSpeed),
        humidity: parseFloat(humidity),
      }),
    });

    // Parse response
    if (response) {
      const data = await response.json();
      adviceDisplay.textContent = data.advice || "No advice available.";
      adviceDisplay.style.color = "green";
    } else {
      throw new Error("Failed to fetch advice.");
    }
  } catch (error) {
    adviceDisplay.textContent = "Error: Unable to fetch advice.";
    adviceDisplay.style.color = "red";
  }
});
