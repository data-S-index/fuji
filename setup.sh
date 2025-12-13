sudo apt update
sudo apt install python3.12-venv tmux

# Create a new environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

