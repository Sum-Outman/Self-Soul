# Self Soul Frontend Application

This is the frontend application for the Self Soul System, built with Vue 3 and Vite.

## Getting Started

### Prerequisites
- Node.js 14+ (recommended: 16+)
- npm 6+ (recommended: 8+)

### Installation

1. Navigate to the app directory:
```bash
cd app
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### For Windows/PowerShell Users
Use the provided PowerShell script to start the development server:
```powershell
# From the root directory
.\start-app.ps1
```

### Manual Start
If you prefer to start manually, run these commands:
```bash
# Navigate to the app directory
cd app

# Start the development server
npm run dev
```

The application will be available at http://localhost:5175

## Project Structure
- `src/` - Source code directory
  - `components/` - Vue components
  - `views/` - View components for routing
  - `utils/` - Utility functions and API services
  - `router/` - Vue Router configuration
  - `App.vue` - Root component
  - `main.js` - Application entry point

## API Configuration
The application connects to the backend services using the following configuration:
- Main API Gateway: http://localhost:8000
- Real-time Stream Manager: http://localhost:8765
- Performance Monitoring: http://localhost:8081

## Available Scripts

### `npm run dev`
Starts the development server with hot-reload.

### `npm run build`
Builds the application for production to the `dist` folder.

### `npm run preview`
Previews the production build locally.

## Notes
- Ensure the Python backend services are running before starting the frontend.
- The frontend uses WebSockets for real-time communication with the backend.
- Configuration for model services ports can be found in `../config/model_services_config.json`