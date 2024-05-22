#!/usr/bin/env bash

cls

echo "Creating datasets for n clients:"
python data/fedexx.py
timeout 3

echo "Start federated learning on n clients:"
python miner.py -g 1 -l 2
timeout 3

for /L %%i in (0,1,1) do (
    echo "Start client %%i"
    python client.py -d "data/federated_data_%%i.d" -e 1
)

timeout 3

python create_csv.py
