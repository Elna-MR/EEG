# Mac Setup Instructions for AI Interview Helper

## Overview
This is an AI-powered interview helper that takes screenshots and sends them to OpenAI for analysis. It's been adapted to work on Mac.

## Setup Steps

### 1. Install Python Dependencies
```bash
cd /Users/elna/Desktop/repo/Self_coder_Elna
pip install -r requirements.txt
```

### 2. Add Your OpenAI API Key
**IMPORTANT**: Edit the file `Avira_Anti_Virus.py` and replace this line:
```python
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # Add your API key here
```

With your actual OpenAI API key:
```python
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```

### 3. Set Up Hotkey Mapping (Choose one method)

#### Method A: Using Hammerspoon (Recommended)
1. Install Hammerspoon: https://www.hammerspoon.org/
2. Run the setup script:
   ```bash
   ./setup_mac_hotkey.sh
   ```
3. Restart Hammerspoon

#### Method B: Using Karabiner-Elements
1. Install Karabiner-Elements: https://karabiner-elements.pqrs.org/
2. Open Karabiner-Elements
3. Go to Complex Modifications → Add rule
4. Create a rule:
   - From: Middle Mouse Button
   - To: Cmd+Shift+X

### 4. Run the Application
```bash
python Avira_Anti_Virus.py
```

## Usage
1. Start the application
2. Enter password: `TahChin2025*`
3. Press `Cmd+Shift+X` (or middle mouse button if mapped) when you want to analyze a question
4. The app will take a screenshot, send it to OpenAI, and display the answer

## Features
- Takes screenshots of interview questions
- Sends to OpenAI GPT-4o for analysis
- Copies answers to clipboard
- Saves logs with timestamps
- 6-hour trial period
- Remote permission checking

## Troubleshooting
- Make sure you have proper permissions for screen recording
- Ensure your OpenAI API key has sufficient credits
- Check that the hotkey isn't conflicting with other applications



