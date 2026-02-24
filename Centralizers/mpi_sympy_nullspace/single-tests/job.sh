#!/bin/bash
if [ -f "report.txt" ]; then
    rm "report.txt"
fi

for ((n = 1; n <= 1; n++));
do
  python3 single_test.py --N $n --K 5
done
python3 read_report.py
#play -n synth 1 sin 440