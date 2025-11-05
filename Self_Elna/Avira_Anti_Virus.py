# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:00:39 2025

@author: saeed
"""

# -*- coding: utf-8 -*-
"""
Avira Anti Virus - Interview Helper App
"""

import time
import base64
import requests
# import keyboard  # Disabled for Mac compatibility
import pyperclip
import mss
from PIL import Image
from io import BytesIO
import traceback
import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
import os
from datetime import datetime, timedelta
import sys
import subprocess
from screeninfo import get_monitors

# --- Icon resource helper for PyInstaller ---
def resource_path(relative_path):
    """ Get absolute path to resource (icon) for dev and PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

ICON_PATH = resource_path("Avira.ico")

# Mac-specific imports and setup
if sys.platform == "darwin":
    # Mac-specific setup can go here if needed
    pass
elif sys.platform == "win32":
    import ctypes
    myappid = 'arasteh.avira.antivirus.2025'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# === CONFIGURATION ===
# TODO: Replace this with your own OpenAI API key
OPENAI_API_KEY = "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"
MODEL = "gpt-4o"
HOTKEY = "f12"  # Use F12 key for Mac compatibility
LOG_DIR = "logs"

# === GUI Display Function ===
def show_solution_gui(text):
    gui = tk.Toplevel()
    gui.title("Avira Anti Virus")
    try:
        gui.iconbitmap(ICON_PATH)
    except tk.TclError:
        pass  # Ignore icon errors
    gui.configure(bg="#1e1e1e")

    try:
        monitors = get_monitors()
        dell_monitor = [m for m in monitors if m.width == 2560 and m.height == 1440][0]
    except Exception:
        dell_monitor = get_monitors()[0]

    width = int(dell_monitor.width * 0.35)
    height = int(dell_monitor.height * 0.8)
    x = dell_monitor.x + dell_monitor.width - width - 30
    y = dell_monitor.y + (dell_monitor.height - height) // 2

    gui.geometry(f"{width}x{height}+{x}+{y}")

    txt = scrolledtext.ScrolledText(
        gui,
        wrap=tk.WORD,
        font=("Consolas", 16),
        bg="#282c34",
        fg="#dcdcdc",
        insertbackground="#ffffff"
    )
    txt.insert(tk.END, text)
    txt.configure(state='disabled')
    txt.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

    tk.Label(
        gui,
        text="✅ Answer copied to clipboard",
        fg="#98c379",
        bg="#1e1e1e",
        font=("Arial", 12)
    ).pack(pady=(0, 15))

    gui.mainloop()

def check_remote_permission():
    # Disabled remote permission check for local use
    return True

# === Screenshot & API Logic ===
def get_mss_monitors():
    with mss.mss() as sct:
        return sct.monitors

def select_question_monitor(monitors):
    """Always select the extended monitor (1920x1080) for coding problems"""
    if len(monitors) <= 1:
        return monitors[0]
    
    # Always look for the extended monitor (1920x1080)
    for i, monitor in enumerate(monitors):
        if monitor['width'] == 1920 and monitor['height'] == 1080:
            return monitor
    
    # If extended monitor not found, fallback to highest resolution
    best_monitor = monitors[0]
    max_area = monitors[0]['width'] * monitors[0]['height']
    
    for monitor in monitors[1:]:
        area = monitor['width'] * monitor['height']
        if area > max_area:
            max_area = area
            best_monitor = monitor
    
    return best_monitor

def take_screenshot(monitor):
    with mss.mss() as sct:
        # First, make sure Chrome is the active window and focus on the tab content
        try:
            import subprocess
            
            # Method 1: Activate Chrome and focus on the active tab
            print("🔄 Activating Chrome window...")
            activate_result = subprocess.run([
                'osascript', '-e', 
                'tell application "Google Chrome" to activate'
            ], capture_output=True, text=True, timeout=2)
            
            if activate_result.returncode == 0:
                print("✅ Chrome activated")
                
                # Wait for Chrome to become active
                import time
                time.sleep(1.0)  # Increased wait time
                
                # Try to get Chrome window bounds
                result = subprocess.run([
                    'osascript', '-e', 
                    'tell application "Google Chrome" to get the bounds of the front window'
                ], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0 and result.stdout.strip():
                    bounds = result.stdout.strip().split(', ')
                    if len(bounds) == 4:
                        x, y, x2, y2 = map(int, bounds)
                        width = x2 - x
                        height = y2 - y
                        
                        print(f"📺 Chrome window bounds: {x}, {y}, {width}x{height}")
                        
                        # Adjust bounds to capture only the tab content (exclude title bar and bookmarks)
                        # Chrome title bar is typically ~28px, bookmarks bar ~40px
                        tab_content_monitor = {
                            'left': x,
                            'top': y + 70,  # Skip title bar and bookmarks
                            'width': width,
                            'height': height - 70  # Reduce height to exclude title bar
                        }
                        
                        print(f"📺 Tab content bounds: {tab_content_monitor}")
                        
                        screenshot = sct.grab(tab_content_monitor)
                        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                        
                        # Save debug image
                        img.save("chrome_tab_capture.png")
                        print(f"📸 Saved Chrome tab capture as chrome_tab_capture.png")
                        
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        return buffered.getvalue(), base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
        except Exception as e:
            print(f"Chrome tab capture failed: {e}")
        
        # Method 2: Try to capture the active window using System Events
        try:
            result = subprocess.run([
                'osascript', '-e', 
                'tell application "System Events" to get the bounds of the front window of the first process whose frontmost is true'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0 and result.stdout.strip():
                bounds = result.stdout.strip().split(', ')
                if len(bounds) == 4:
                    x, y, x2, y2 = map(int, bounds)
                    width = x2 - x
                    height = y2 - y
                    
                    print(f"📺 Active window bounds: {x}, {y}, {width}x{height}")
                    
                    window_monitor = {
                        'left': x,
                        'top': y, 
                        'width': width,
                        'height': height
                    }
                    
                    screenshot = sct.grab(window_monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    
                    # Save debug image
                    img.save("active_window_capture.png")
                    print(f"📸 Saved active window capture as active_window_capture.png")
                    
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    return buffered.getvalue(), base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
        except Exception as e:
            print(f"Active window capture failed: {e}")
        
        # Method 3: Fallback to monitor capture
        print(f"📺 Falling back to monitor capture: {monitor}")
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        
        # Save debug image
        img.save("monitor_capture.png")
        print(f"📸 Saved monitor capture as monitor_capture.png")
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue(), base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_solution(image_base64):
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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === Save Logs ===
def save_log(image_bytes, solution):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(LOG_DIR, f"log_{timestamp}")

    img_path = f"{base_path}.png"
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    txt_path = f"{base_path}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(solution)

def self_destruct():
    exe_path = sys.argv[0]
    bat_path = exe_path + ".bat"
    with open(bat_path, "w") as f:
        f.write(f"""
        @echo off
        timeout /t 2 > nul
        del "{exe_path}"
        del "{bat_path}"
        """)
    subprocess.Popen(bat_path, shell=True)

# === GUI Application ===
class AIInterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Avira Anti Virus")
        try:
            self.root.iconbitmap(ICON_PATH)
        except tk.TclError:
            pass  # Ignore icon errors
        self.root.geometry("400x320")
        self.root.configure(bg="#1e1e1e")

        # Time limit check disabled for easier use
        # self.start_time = self.load_start_time()
        # if not self.check_time_limit():
        #     messagebox.showerror("Expired", "⏳ Trial period has expired. This app will now delete itself.")
        #     self.root.after(500, self_destruct)
        #     self.root.destroy()
        #     return

        if not self.check_remote_permission():
            messagebox.showerror("Access Denied", "🚫 Remote access disabled by administrator.")
            self.root.after(500, self_destruct)
            self.root.destroy()
            return

        # Password check disabled for easier use
        # if not self.check_password():
        #     messagebox.showerror("Authentication Failed", "❌ Incorrect password.")
        #     self.root.destroy()
        #     return

        self.label = tk.Label(self.root, text="🟢 AI Assistant Ready - Middle mouse button to analyze",
                              fg="#98c379", bg="#1e1e1e", font=("Arial", 12))
        self.label.pack(pady=(15, 10))

        # Timer disabled since time limit is removed
        # self.time_label = tk.Label(self.root, text="", fg="#ffcc00",
        #                            bg="#1e1e1e", font=("Arial", 12))
        # self.time_label.pack(pady=(0, 10))
        # self.update_timer()

        self.status_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            height=10,
            font=("Consolas", 10),
            bg="#282c34",
            fg="#dcdcdc"
        )
        self.status_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.status_text.insert(tk.END, "Ready to assist...\n")
        self.status_text.configure(state='disabled')

        self.monitors = get_mss_monitors()
        self.question_monitor = select_question_monitor(self.monitors)
        self.update_status(f"📺 Always using extended monitor: {self.question_monitor}")
        self.update_status(f"🎯 Ready to capture NeetCode problems!")
        # For Mac, we'll use a simple button and keyboard shortcut
        if sys.platform == "darwin":
            # Create a button for manual triggering
            self.trigger_button = tk.Button(
                self.root, 
                text="🎯 Middle Mouse to Analyze Screen", 
                command=self.on_hotkey_mac,
                bg="#4CAF50",
                fg="white",
                font=("Arial", 12, "bold")
            )
            self.trigger_button.pack(pady=10)
            
            # Bind F15 for middle mouse button
            self.root.bind('<F15>', lambda e: self.on_hotkey_mac())
            self.root.focus_set()  # Make sure the window can receive key events
        else:
            # Windows hotkey handling - disabled for Mac compatibility
            pass

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def check_remote_permission(self):
        # Disabled remote permission check for local use
        return True

    def load_start_time(self):
        try:
            with open("start_time.txt", "r") as f:
                return datetime.fromisoformat(f.read().strip())
        except Exception:
            now = datetime.now()
            with open("start_time.txt", "w") as f:
                f.write(now.isoformat())
            return now

    def check_time_limit(self):
        return datetime.now() - self.start_time <= timedelta(hours=6)

    def update_timer(self):
        elapsed = datetime.now() - self.start_time
        remaining = max(timedelta(0), timedelta(hours=6) - elapsed)
        hours, rem = divmod(remaining.seconds, 3600)
        minutes, _ = divmod(rem, 60)
        self.time_label.config(text=f"⏳ Time remaining: {hours}h {minutes}m")
        if remaining.total_seconds() <= 0:
            messagebox.showinfo("Time Expired", "⏰ Your time has ended. This app will now close.")
            self.root.after(1000, self_destruct)
            self.root.destroy()
        else:
            self.root.after(60000, self.update_timer)

    def check_password(self):
        password = simpledialog.askstring("Password", "Enter password:", show='*', parent=self.root)
        return password == "TahChin2025*"

    def update_status(self, message):
        self.status_text.configure(state='normal')
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.configure(state='disabled')
        self.status_text.see(tk.END)

    def on_hotkey_mac(self):
        """Middle mouse button handler - take screenshot and show AI analysis"""
        print("🎯 MIDDLE MOUSE BUTTON TRIGGERED!")
        self.update_status("🎯 Middle mouse detected - taking screenshot...")
        # Process immediately when middle mouse is pressed
        self.root.after(50, self.process_screenshot)

    def on_hotkey(self, e):
        # Disabled for Mac compatibility
        pass

    def process_screenshot(self):
        print("📸 PROCESS_SCREENSHOT FUNCTION CALLED!")
        try:
            # Take screenshot silently
            print("📸 Taking screenshot...")
            image_bytes, image_base64 = take_screenshot(self.question_monitor)
            print("📸 Screenshot taken successfully!")
            self.update_status("📤 Analyzing with AI...")
            
            # Get AI solution
            print("🤖 Calling AI API...")
            solution = get_solution(image_base64)
            print("🤖 AI response received!")
            
            # Copy to clipboard silently
            pyperclip.copy(solution)
            print("📋 Copied to clipboard")
            
            # Save log silently
            save_log(image_bytes, solution)
            print("💾 Log saved")
            
            self.update_status("✅ Analysis complete!")
            
            # Always show the AI analysis popup
            print("🪟 Showing AI analysis popup...")
            self.root.after(0, lambda: show_solution_gui(solution))
            
        except Exception as e:
            print(f"❌ ERROR in process_screenshot: {str(e)}")
            error_msg = f"❌ Error: {str(e)}"
            self.update_status(error_msg)
            # Show error in popup too
            self.root.after(0, lambda: show_solution_gui(f"Error occurred: {str(e)}"))
        
        # Clean up
        self.root.after(100, lambda: None)

    def on_closing(self):
        self.root.destroy()

# === Main Runner ===
if __name__ == "__main__":
    root = tk.Tk()
    app = AIInterviewApp(root)
    root.mainloop()
