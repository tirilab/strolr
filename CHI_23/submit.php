<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = $_POST['name'];
    $age = $_POST['age'];
    $feedback = $_POST['feedback'];

    // Store the survey data in a database or file (implement your data storage logic here)

    // You can redirect the user to a thank you page or display a confirmation message
    // header("Location: thank_you.php");
    echo "Thank you for submitting the survey!";
}
?>
