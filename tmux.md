tmux new -s main

source venv/bin/activate

python3 fill-database-fuji.py --threads 75

tmuxt list-sessions
tmux ls

tmux attach # attach to the last session
tmux attach -t main

tmux kill-session -t main
