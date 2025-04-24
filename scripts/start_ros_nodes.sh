#!/bin/sh
tmux new-session -d -s ros -n roscore
tmux send-keys -t ros "roscore" Enter
tmux new-window -t ros:1 -n vicon "roslaunch vicon_bridge vicon.launch"
tmux new-window -t ros:2 -n echoer "conda activate jaxrl5; python3 redexp/envs/turtlebot.py"
tmux attach -t ros
