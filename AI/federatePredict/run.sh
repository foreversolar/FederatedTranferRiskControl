python3 1.py
sleep 2
python3 -u 3.py >server0.log &
sleep 2
python3 -u 2.py 0 >trainer0.log &
sleep 2
python3 -u 2.py 1 >trainer1.log &
