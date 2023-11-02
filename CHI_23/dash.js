// Simulated wearable data (replace with actual data retrieval)
const wearableData = {
    heartRate: 75,
    steps: 5000,
    caloriesBurned: 250,
};

// Function to update the dashboard with wearable data
function updateDashboard() {
    const dashboard = document.getElementById("dashboard");

    const dataHTML = `
        <h2>Wearable Data</h2>
        <ul>
            <li>Heart Rate: ${wearableData.heartRate} bpm</li>
            <li>Steps: ${wearableData.steps}</li>
            <li>Calories Burned: ${wearableData.caloriesBurned} cal</li>
        </ul>
    `;

    dashboard.innerHTML = dataHTML;
}

// Call the updateDashboard function to display initial data
updateDashboard();

// Simulate updating data every 10 seconds (replace with real data retrieval)
setInterval(() => {
    // Update wearable data (e.g., fetch from an API)
    // For now, we'll just increment the values for demonstration purposes
    wearableData.heartRate += 1;
    wearableData.steps += 100;
    wearableData.caloriesBurned += 5;

    // Update the dashboard
    updateDashboard();
}, 10000); // Update every 10 seconds
