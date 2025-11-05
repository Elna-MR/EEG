#NoTrayIcon           ; Hides the green tray icon
DetectHiddenWindows, On
SetTitleMatchMode, 2

; Hide the script window from taskbar
WinHide, % "ahk_class AutoHotkey"

; Remap Middle Mouse Button to Ctrl+Shift+X
MButton::
Send ^+x
return
