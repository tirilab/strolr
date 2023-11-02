<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = $_POST['username'];
    $password = $_POST['password'];

    // Perform authentication (replace with your authentication logic)
    if ($username === 'your_username' && $password === 'your_password') {
        // Redirect to a dashboard page or perform other actions
        header("Location: dashboard.php");
    } else {
        // Authentication failed, display an error message
        echo "Login failed. Please check your username and password.";
    }
}
?>
