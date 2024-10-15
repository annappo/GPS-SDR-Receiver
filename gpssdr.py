import os
import subprocess
from src.gpsglob import \
    SRC_PATH,LOAD_PICKLE,SAVE_PICKLE,SEND_OVER_UDP

if LOAD_PICKLE or SEND_OVER_UDP:
    gpseval = os.path.join(SRC_PATH,'gpseval.py')
    sub_eval = subprocess.Popen(['python',gpseval])
    print(f'GPS evaluation started (PID {sub_eval.pid})')

if not LOAD_PICKLE and (SAVE_PICKLE or SEND_OVER_UDP):
    gpsrecv = os.path.join(SRC_PATH,'gpsrecv.py')
    sub_recv = subprocess.Popen(['python',gpsrecv])
    print(f'GPS receiver started (PID {sub_recv.pid})')
