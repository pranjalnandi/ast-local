#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Create a Python virtual environment using Python 3.10
echo "Creating Python 3.10 virtual environment..."
python3 -m venv astvenv

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source astvenv/bin/activate

# 3. Upgrade pip to the latest version
# echo "Upgrading pip..."
# pip install --upgrade pip

# 4. Install libraries from wheel files in the specified directory
WHEEL_DIR="./whl_files"  # Change this if your .whl files are in a different folder
if [ -d "$WHEEL_DIR" ]; then
  echo "Installing wheel files from $WHEEL_DIR..."
  for file in "$WHEEL_DIR"/*.whl; do
    if [ -f "$file" ]; then
      echo "Installing $file"
      pip install "$file"
    fi
  done
else
  echo "Wheel directory '$WHEEL_DIR' does not exist. Skipping wheel installation."
fi


# 5. Install system packages using apt-get
echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# 6. Install Python packages from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
  echo "Installing Python packages from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Skipping installation of requirements.txt packages."
fi

pip install numpy==1.23.5

echo "Setup completed successfully."
