function toggleSubtopics(categoryId) {
    const subtopics = document.getElementById(categoryId);
    const allSubtopics = document.querySelectorAll('.subtopics');

    // Hide all subtopics except the one with the provided categoryId
    allSubtopics.forEach((subtopic) => {
        if (subtopic.id !== categoryId) {
            subtopic.style.display = "none";
        }
    });

    // Toggle the display of subtopics for the clicked category
    if (subtopics.style.display === "block") {
        subtopics.style.display = "none";
    } else {
        subtopics.style.display = "block";
    }
}
