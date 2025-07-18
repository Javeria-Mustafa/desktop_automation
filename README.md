Desktop Use is an advanced AI-powered desktop automation framework that enables Large Language Models (LLMs) to interact with desktop applications through natural language instructions. The framework combines computer vision, UI element detection, and intelligent action planning to automate complex desktop workflows.
🚀 Features
Core Capabilities

AI-Driven Automation: Uses OpenAI GPT models to understand and execute complex desktop tasks
Visual Understanding: Takes screenshots and analyzes UI elements for context-aware automation
Cross-Application Support: Works with any Windows desktop application
Natural Language Interface: Execute tasks using simple English instructions
Robust Error Handling: Built-in retry mechanisms and fallback strategies
Session Recording: Generate GIFs and detailed logs of automation sessions

Advanced Features

UI Element Detection: Automatic identification of interactive elements using Windows UI Automation
Smart Click Strategies: Multiple fallback methods for reliable element interaction
Form Automation: Intelligent form filling with data validation
File Operations: Automated file opening, saving, and management
Menu Navigation: Automatic menu detection and keyboard shortcut usage
Validation System: Optional output validation to ensure task completion

Monitoring & Debugging

Comprehensive Logging: Detailed step-by-step execution logs
Telemetry: Anonymous usage analytics for framework improvement
Screenshot Capture: Visual documentation of each automation step
Error Tracking: Detailed error reporting and debugging information

📋 Requirements

Operating System: Windows 10/11
Python: 3.8 or higher
Memory: 4GB RAM minimum (8GB recommended)
Display: 1920x1080 minimum resolution

🛠 Installation
1. Clone the Repository
bashgit clone https://github.com/Javeria-Mustafa/desktop_automation/desktop-use.git
cd desktop-use
2. Create Virtual Environment
bashpython -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
3. Install Dependencies
bashpip install -r requirements.txt
4. Environment Configuration
Create a .env file in the root directory:
envOPENAI_API_KEY=your_openai_api_key_here
ANONYMIZED_TELEMETRY=true
DESKTOP_USE_LOGGING_LEVEL=info
