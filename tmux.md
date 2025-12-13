tmux new -s main

source venv/bin/activate

python3 fill-database-fuji.py --threads 60
