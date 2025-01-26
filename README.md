# GeneCounsel-AI

GeneCounsel-AI is an AI-powered educational platform designed to help pharmacy students practice pharmacogenetic counseling across multiple languages and therapeutic areas. This guide will help you set up and run the application locally on your computer.

## Prerequisites

Before you begin, ensure you have the following installed on your computer:

1. **Python 3.8 or higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, make sure to check "Add Python to PATH"

2. **Git** (for downloading the code)
   - Download from [git-scm.com](https://git-scm.com/downloads)

3. **OpenAI API Key**
   - Sign up at [OpenAI](https://platform.openai.com/)
   - Create an API key from your dashboard

## Installation

1. **Clone the Repository**
   Open a terminal (Command Prompt on Windows) and run:
   ```bash
   git clone https://github.com/yourusername/genecounsel-ai.git
   cd genecounsel-ai
   ```

2. **Create a Virtual Environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Your OpenAI API Key**
   Create a `.streamlit` folder in the project directory and create a `secrets.toml` file inside it:
   ```bash
   # On Windows
   mkdir .streamlit
   echo openai.api_key = "your-api-key-here" > .streamlit/secrets.toml

   # On macOS/Linux
   mkdir .streamlit
   echo "openai.api_key = \"your-api-key-here\"" > .streamlit/secrets.toml
   ```
   Replace "your-api-key-here" with your actual OpenAI API key.

## Running the Application

1. **Start the Application**
   ```bash
   streamlit run src/gc18.py
   ```

2. **Access the Application**
   - The application will automatically open in your default web browser
   - If it doesn't, open your browser and go to `http://localhost:8501`

## Using GeneCounsel-AI

### Available Features

The platform includes four therapeutic modules:
- Cardiovascular
- Epilepsy 
- HIV
- Psychiatric care

Each module provides:
- Dynamic patient scenarios
- Pharmacogenetic test results
- Standardized assessment framework
- Real-time feedback

### Supported Languages

The platform supports 15 languages which can be selected from the sidebar:
- English
- Spanish (Español)
- Arabic (العربية)
- Vietnamese (Tiếng Việt)
- Mandarin (中文)
- Urdu (اردو)
- Hungarian (Magyar)
- Hindi (हिंदी)
- Telugu (తెలుగు)
- Filipino (Tagalog)
- Korean (한국어)
- Russian (Русский)
- French (Français)
- Portuguese (Português)
- Japanese (日本語)

### Basic Navigation

1. **Select Exercise**: Choose a therapeutic module from the sidebar dropdown
2. **Choose Language**: Select your preferred language from the sidebar
3. **Start Conversation**: Type your messages in the chat input at the bottom
4. **Get Evaluation**: Click "Evaluate Conversation" in the sidebar when ready
5. **Save Conversation**: Use "Download Conversation" to save your session

### Assessment Framework

The system evaluates performance across 10 areas:
1. Professional Communication
2. Empathy and Rapport
3. Medical Language Appropriateness
4. Technical Accuracy
5. Information Collection
6. Pharmacogenetic Test Explanation
7. Results Interpretation
8. Medication Impact Communication
9. Follow-up Guidance
10. Session Management

Each category is scored 0-4 points, totaling 40 possible points.

## Troubleshooting

Common issues and solutions:

1. **Application Won't Start**
   - Ensure Python is properly installed and in your PATH
   - Verify all requirements are installed: `pip list`
   - Check if port 8501 is available

2. **API Key Issues**
   - Verify your API key is correctly placed in `.streamlit/secrets.toml`
   - Check OpenAI account status and billing
   - Ensure no spaces or quotes around the API key

3. **Language Display Problems**
   - Ensure your system has appropriate language fonts installed
   - Try updating your web browser
   - Clear browser cache and reload

For additional help, please create an issue in the GitHub repository.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.

Copyright © 2025 Dayanjan S. Wijesinghe

## Contact

For questions or support, contact:

Dayanjan S. Wijesinghe, Ph.D.  
Department of Pharmacotherapy and Outcome Sciences  
School of Pharmacy, Virginia Commonwealth University  
Email: wijesinghes@vcu.edu
