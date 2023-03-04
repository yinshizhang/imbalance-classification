#!/bin/bash
# This script is used to approximate the time of the fraud

# cd to the directory where the script is located
cd "$(dirname "$0")"

for i in `seq 1 4`; do
    echo "Running $i"
    nohup python time_aprox.py $i > time_aprox_$i.log 2>&1 &
done

# 2:65
# 4:108
# 8:200

# chose 10