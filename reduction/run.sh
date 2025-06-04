#!/bin/bash

declare -a pids

for i in {0..31}
do
  python main.py -i "$i" -b 32 &
  pids+=($!)
done

for pid in "${pids[@]}"
do
  wait "$pid"
done

python main.py -m 32

echo "all done"