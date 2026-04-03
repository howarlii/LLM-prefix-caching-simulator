#!/bin/bash

python experiments/run_page_size.py --dataset  sharegpt_90k_raw  --capacity 20 --ordering random --strategy lru
python experiments/run_page_size.py --dataset  sharegpt_90k_raw  --capacity 40 --ordering random --strategy lru
python experiments/run_page_size.py --dataset  sharegpt_90k_raw  --capacity 80 --ordering random --strategy lru

