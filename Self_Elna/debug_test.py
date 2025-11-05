#!/usr/bin/env python3
"""
Debug version to test screenshot and API functionality
"""

import mss
from PIL import Image
from io import BytesIO
import base64
import requests
import tkinter as tk
from tkinter import messagebox
import traceback

# Your API key
OPENAI_API_KEY = "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"
MODEL = "gpt-4o"

def get_monitors():
    """Get available monitors"""
    with mss.mss() as sct:
        return sct.monitors

def select_extended_monitor(monitors):
    """Select the extended monitor (1920x1080)"""
    for monitor in monitors:
        if monitor['width'] == 1920 and monitor['height'] == 1080:
            return monitor
    return monitors[0]  # Fallback

def take_screenshot(monitor):
    """Take screenshot of specified monitor"""
    print(f"📸 Taking screenshot of monitor: {monitor}")
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        
        # Save for debugging
        img.save("debug_screenshot.png")
        print(f"✅ Screenshot saved as debug_screenshot.png")
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return image_base64

def call_openai_api(image_base64):
    """Call OpenAI API with the screenshot"""
    print("🤖 Calling OpenAI API...")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "You are a world-class AI assistant helping a candidate solve interview questions from top tech companies. "
        "Analyze the screenshot carefully and look for any coding problems, algorithm questions, or programming challenges. "
        "If you find a coding question, provide the solution. If you don't see a clear coding question, explain what you see and ask for clarification.\n\n"
        "When you find a coding question:\n"
        "1. Extract the complete problem statement\n"
        "2. Identify the problem type (array, string, tree, graph, etc.)\n"
        "3. Provide a clear, concise solution with explanation\n"
        "4. Use the appropriate programming language (Python preferred)\n"
        "5. Include time/space complexity if relevant\n\n"
        "If no coding question is visible, describe what you see in the image and ask the user to navigate to a coding problem."
    )
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print("✅ API call successful!")
            return answer
        else:
            error_msg = f"❌ API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ Exception: {str(e)}"
        print(error_msg)
        return error_msg

def show_result(result):
    """Show the result in a popup window"""
    print("🪟 Showing result window...")
    
    root = tk.Tk()
    root.title("AI Analysis Result")
    root.geometry("800x600")
    
    # Create text widget
    text_widget = tk.Text(root, wrap=tk.WORD, font=("Consolas", 12))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Insert result
    text_widget.insert(tk.END, result)
    text_widget.configure(state='disabled')
    
    # Add close button
    close_btn = tk.Button(root, text="Close", command=root.destroy)
    close_btn.pack(pady=10)
    
    root.mainloop()

def test_full_process():
    """Test the complete process"""
    print("🚀 Starting debug test...")
    
    try:
        # Step 1: Get monitors
        monitors = get_monitors()
        print(f"📺 Found {len(monitors)} monitors")
        
        # Step 2: Select extended monitor
        monitor = select_extended_monitor(monitors)
        print(f"🎯 Selected monitor: {monitor}")
        
        # Step 3: Take screenshot
        image_base64 = take_screenshot(monitor)
        print(f"📸 Screenshot taken, base64 length: {len(image_base64)}")
        
        # Step 4: Call API
        result = call_openai_api(image_base64)
        print(f"🤖 API result length: {len(result)}")
        
        # Step 5: Show result
        show_result(result)
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        error_msg = f"❌ Test failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        messagebox.showerror("Debug Test Failed", error_msg)

if __name__ == "__main__":
    test_full_process()






