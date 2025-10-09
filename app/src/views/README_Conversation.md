# Conversation Component

The Conversation component is a comprehensive chat interface that allows users to interact with various AI models in the Self Soul system. It provides a unified and user-friendly way to communicate with different models, view model status information, and manage conversations.

## Features

- **Model Selection**: Switch between different AI models (Language Model, Management Model, and From Scratch Model)
- **Real-time Chat**: Send messages and receive responses from AI models
- **Model Status Display**: View detailed information about the selected model's status
- **Connection Monitoring**: Real-time display of backend connection status
- **Message Management**: Clear all messages to start a new conversation
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Loading Indicators**: Visual feedback during message processing

## Technical Details

### Frontend Implementation
- Built with Vue 3 and Composition API
- Uses a clean black-and-white-gray color scheme with light theme
- Implements component-based architecture for maintainability
- Includes comprehensive error handling and fallback mechanisms

### Backend Integration
- Connects to the main API endpoint `/api/chat` for language models
- Uses specialized endpoint `/api/models/8001/chat` for management model
- Retrieves model status through `/api/models/{model_type}/status` endpoints
- Monitors backend health with periodic checks to `/api/health`

### Message Flow
1. User sends a message through the input field
2. Message is added to the chat history
3. Loading indicator is shown while processing
4. Request is sent to the appropriate API endpoint based on selected model
5. Response is received and displayed in the chat interface
6. Conversation history is updated and maintained

## Usage

1. Navigate to the Conversation page using the top navigation menu
2. Select the desired AI model from the dropdown menu
3. Type your message in the input field and press Send or Enter
4. Wait for the AI response to appear in the chat interface
5. Use the Clear All Messages button to reset the conversation
6. Toggle the model status panel to view detailed information about the selected model

## Screenshots

### Desktop View
- Full-featured interface with model selection, status panel, chat container, and input area
- Optimized layout for larger screens with maximum content visibility

### Mobile View
- Responsive design that adapts to smaller screens
- Stacked elements for better touch interaction
- Simplified controls for mobile users

## Development Notes

- The component uses mock data for demonstration purposes when the backend is not available
- All API calls are wrapped in error handling to ensure graceful degradation
- The conversation history is limited to 50 messages to optimize performance
- Connection status is checked periodically every 5 seconds

## Compatibility

- Works with all modern web browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled for full functionality
- Supports screen readers and keyboard navigation

## Future Enhancements
- Support for multimodal inputs (images, audio, video)
- Conversation persistence across sessions
- Advanced formatting options for messages
- User preferences customization
