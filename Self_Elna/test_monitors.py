#!/usr/bin/env python3
"""
Test script to show available monitors and test screenshot functionality
"""

import mss
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import os

def test_monitors():
    """Test and display available monitors"""
    print("🔍 Testing monitors...")
    
    with mss.mss() as sct:
        monitors = sct.monitors
        print(f"\n📺 Found {len(monitors)} monitor(s):")
        
        for i, monitor in enumerate(monitors):
            print(f"  Monitor {i}: {monitor}")
            print(f"    - Size: {monitor['width']}x{monitor['height']}")
            print(f"    - Position: ({monitor['left']}, {monitor['top']})")
            print()
    
    return monitors

def test_screenshot(monitor_index=0):
    """Test screenshot on specific monitor"""
    print(f"📸 Testing screenshot on monitor {monitor_index}...")
    
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            print(f"❌ Monitor {monitor_index} not found!")
            return None
        
        monitor = monitors[monitor_index]
        print(f"📺 Capturing from monitor {monitor_index}: {monitor}")
        
        try:
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            
            # Save screenshot for testing
            filename = f"test_screenshot_monitor_{monitor_index}.png"
            img.save(filename)
            print(f"✅ Screenshot saved as: {filename}")
            
            # Show image info
            print(f"📏 Image size: {img.size}")
            print(f"🎨 Image mode: {img.mode}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            return None

def show_monitor_info():
    """Show monitor information in a GUI"""
    root = tk.Tk()
    root.title("Monitor Test")
    root.geometry("600x400")
    
    monitors = test_monitors()
    
    # Create text widget to show monitor info
    text_widget = tk.Text(root, wrap=tk.WORD, height=20)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    info_text = f"Found {len(monitors)} monitor(s):\n\n"
    for i, monitor in enumerate(monitors):
        info_text += f"Monitor {i}:\n"
        info_text += f"  Size: {monitor['width']}x{monitor['height']}\n"
        info_text += f"  Position: ({monitor['left']}, {monitor['top']})\n"
        info_text += f"  Full info: {monitor}\n\n"
    
    text_widget.insert(tk.END, info_text)
    text_widget.configure(state='disabled')
    
    # Add buttons to test screenshots
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    for i in range(len(monitors)):
        btn = tk.Button(
            button_frame, 
            text=f"Test Monitor {i}", 
            command=lambda idx=i: test_screenshot_gui(idx)
        )
        btn.pack(side=tk.LEFT, padx=5)
    
    def test_screenshot_gui(monitor_index):
        filename = test_screenshot(monitor_index)
        if filename:
            messagebox.showinfo("Success", f"Screenshot saved as: {filename}")
        else:
            messagebox.showerror("Error", "Failed to take screenshot")
    
    root.mainloop()

if __name__ == "__main__":
    print("🚀 Starting monitor test...")
    show_monitor_info()






