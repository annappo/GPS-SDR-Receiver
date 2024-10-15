# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

from datetime import datetime
import numpy as np
from os.path import dirname,abspath,join,normpath

# ========== global variables ===============================
# are imported by 
# 1: gpsrecv.py, 2: gpseval.py, 3: gpsui.py, and 4: gpslib.py 

MEAS_TIME = 6000           # (1) in s; gpsrecv is closed after this period
LIVE_MEAS = False          # (1) True for real-time navigation 
BIN_DATA = 'test.bin'      # (1) filename for IQ data if LIVE_MEAS=False
START_STREAM = 0           # (1) in units of stream length (NGPS bytes);
                           # may be used to skip beginning of BIN_DATA
SAVE_TRACK = False         # (2) save track of fixed positions as gpx file 
SAVE_EVAL_RES = False      # (2) save results of gpseval in JSON files:
                           # code phases, subframes, satellite infos, 
                           # and position fixes                           
SAVE_PICKLE = False        # (1) save results of gpsrecv in pickle file;
                           # filename is f'{SAVE_DATE}_gpsResult.pickle)'
LOAD_PICKLE = False        # (2) evaluate pickle file instead of live data
PICKLE_FILE = ''           # (2) data to use for evaluation 
SEND_OVER_UDP = True       # (1) can be set False if SAVE_PICKLE = True
SAVE_DATE = datetime.now().strftime("%y%m%d_%H%M%S")  
                           # (1,2) used in filenames for saving results
REL_PATH = '../data'       # relative path from 'src' to data folder
                           # used for loading and saving data
SRC_PATH = dirname(abspath(__file__))
DATA_PATH = join(normpath(join(SRC_PATH,REL_PATH)),'')  # (1,2) 

# ------- device dependent parameters ---------------------------

MIN_SAT = 4                # (2) minimum number of satellites for positioning
                           # typically 4; 3 is possible if CONF_HEIGHT is set
                           # True in user interface (not recommended)
MAX_SAT = 11               # (1) limits number of used satellites, e.g. if 
                           # computer is slow 
SDR_FREQCORR = -2          # (1) in ppm; frequency correction of rtl-sdr stick
IT_SWEEP = 40              # (1) for search of single satellite in SatStream()
IT_SWEEP_ALL = 10          # (1) for global search of satellites
                           # both IT values should be ok even for older 
                           # computers; limit avoids (harmless) data skips
                           
# ------- positioning ---------------------------------------------

POS_AVG_IN_SEC = 1         # (2) in s; time for averaging mean position; 
                           # (2) 0 corresponds to continuous averaging                 
MIN_OUT_DIST = 500         # (2) in m;  for identifying outliers
MAX_SPEED = 60             # (2) in m/s; 216 km/h; to determine outliers
CONF_HEIGHT = False        # (2) if True HEIGHT is used as another boundary
                           # condition for positioning
HEIGHT = 0                 # (2) in m; GPS height (not mean sea level); 
                           # (2) both values can be set also in UI 
HEIGHT_DEV = 10            # (2) in m; standard deviation of HEIGHT
LSF_MAX_IT = 15            # (2) max number of iteration for LSF
MAX_RESIDUAL = 1.0E-07     # (2) min deviation for LSF
LSF_WEIGHT = True          # (2) use weight function in LSF 

# -------- C/A code correlation ----------------------------------

CORR_AVG = 8                # (1) passed to SatStream - max is N_CYC
                            # average number of FFTs for C/A code correlation
CORR_MIN = 8                # (1) required amplitude of correlation peak 
                            # in units of standard deviations
SWEEP_CORR_AVG = 4          # (1) similar to CORR_AVG; speeds up calculation if 
                            # smaller; precision is less important for sweep

# ---------- sweep ----------------------------------------------

MIN_FREQ = -5000.0          # (1,4) in Hz; lowest possible doppler frequency
MAX_FREQ = +5000.0          # (1,4) in Hz; highest possible doppler frequncy
STEP_FREQ = 200             # (1,4) in Hz; step size for searching satellite's 
                            # doppler frequencies
                            
# ------- UDP settings ------------------------------------------

UDP_IP = '127.0.0.1'       # (1) IP address of computer running gpseval;
                           # default is local (127.0.0.1)
                           # might require firewall settings if not local
UDP_PORT = 61431           # (1,2) port for sending to gpseval 
UDP_PORT_2 = UDP_PORT + 1  # (1,2) port for sending to gpsrecv
                           # ports in range 49152 to 65536 are free to use 
UDP_BUFSIZE_1 = 65504      # (2) max payload for udp messages 
                           # 11 sats & N_CYC=32: max pickle size is ~14000 
                           # 11 sats & N_CYC=16: max pickle size is ~21000
UDP_BUFSIZE_2 = 1024       # (1) buffer size for single commands
UDP_ALL = ''               # (2) accept all IP addresses
UDP_WAIT_TIME = 0.1        # (2) in seconds 

# --------- ephemerides ------------------------------------------

SAVE_EPHEM = True          # (2) ephemerides collected in a first run can be 
LOAD_EPHEM = True          # used for following measurements; valid for ~2h  
EPHEM_FILE = 'gpsEphem.json'                          
                           # (2) filename of saved ephemerides 

# ------ user interface  -------------------------------------------

SHOW_MAP_HTML = DATA_PATH + 'showMap.html'
                           # (3) path and filename for saved map & track
TRACK_COL = 'red'          # (3) color of track on map
TRACK_WEIGHT = 3           # (3) line thickness on map
MARKER_SIZE = 10           # (3) size of marker for last position on map 
MARKER_COL = 'blue'        # (3) color of marker on map
TRACK2_COL = 'blue'        # (3) color of track 2 on map
TRACK2_WEIGHT = 1          # (3) line 2 thickness on map 
SCALE_THRES = 1500         # (3) in m; threshold for scaling of graph in km

FIG_SIZE = (9,7)           # (2) in inch; initial size of canvas
PLOT_SCALE = 500           # (2) in m; scale of graph for track; 
                           # (2) plot initially set to -PLOT_SCALE..+PLOT_SCALE
POS_SIZE = 1.0             # (2) size of marker for single position on track

# ---- system parameters ---------------------------------------------
# The parameters below should only be changed very cautiously.

CODE_SAMPLES = 2048         # (1,2,4) length of interpolated C/A codes;
                            # only powers of 2 due to FFT; do not change! 
SAMPLE_RATE = 1000*CODE_SAMPLES  # (1,2,4) samples per second
N_CYC = 32                  # (1,2,4) currently possible are (32,16,8);
                            # 8 needs fast processor & many cores
                            # adjust IT_SWEEP, IT_SWEEP_ALL accordingly
NGPS = N_CYC*CODE_SAMPLES   # (1,2,4) number of samples for each reading

# ======== floating point types used in arrays =======================
# 32-bit types overall ~1.4 times faster than 64-bit

MY_FLOAT = np.float32       # (1,2,4)
MY_COMPLEX = np.complex64   # (1,2,4)



