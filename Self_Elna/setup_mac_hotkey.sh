#!/bin/bash
# Mac Middle Mouse Button to Hotkey Script
# This script uses Hammerspoon to remap middle mouse button to Cmd+Shift+X

echo "Setting up Mac hotkey mapping..."

# Check if Hammerspoon is installed
if ! command -v hammerspoon &> /dev/null; then
    echo "Hammerspoon is not installed. Please install it first:"
    echo "1. Download from: https://www.hammerspoon.org/"
    echo "2. Install the app"
    echo "3. Run this script again"
    exit 1
fi

# Create Hammerspoon config directory if it doesn't exist
mkdir -p ~/.hammerspoon

# Create the Hammerspoon configuration
cat > ~/.hammerspoon/init.lua << 'EOF'
-- Middle Mouse Button to Cmd+Shift+X mapping
hs.hotkey.bind({}, "middlemouse", function()
    hs.eventtap.keyStroke({"cmd", "shift"}, "x")
end)

-- Show notification when script loads
hs.notify.new({title="Hammerspoon", informativeText="Middle mouse button mapped to Cmd+Shift+X"}):send()
EOF

echo "Hammerspoon configuration created!"
echo "Please restart Hammerspoon or reload the configuration."
echo ""
echo "Alternative method using Karabiner-Elements:"
echo "1. Install Karabiner-Elements from: https://karabiner-elements.pqrs.org/"
echo "2. Add this rule in Complex Modifications:"
echo "   - From: Middle Mouse Button"
echo "   - To: Cmd+Shift+X"


