function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') {
        return;
    }

    // Clear user input
    document.getElementById('user-input').value = '';

    // Display user message in chat box
    appendMessage('user', userInput);

    // Send the user input to the Flask API
    fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: userInput
        })
    })
        .then(response => response.json())
        .then(data => {
            // Display the chatbot response in chat box
            appendMessage('chatbot', data.answer);

            // Display suggestions
            displaySuggestions(data.suggestions);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function displaySuggestions(suggestions) {
    const suggestionsContainer = document.getElementById('suggestions');
    suggestionsContainer.innerHTML = '';

    suggestions.forEach(suggestion => {
        const suggestionButton = document.createElement('button');
        suggestionButton.textContent = suggestion;
        suggestionButton.addEventListener('click', () => {
            // Handle suggestion click event: set suggestion as user input and send it
            document.getElementById('user-input').value = suggestion;
            sendMessage();
        });

        suggestionsContainer.appendChild(suggestionButton);
    });
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender);
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    // Scroll to the bottom of the chat box
    chatBox.scrollTop = chatBox.scrollHeight;
}