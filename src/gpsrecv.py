# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

from gpsglob import *
import asyncio
import numpy as np
from scipy.fft import fft, ifft
import time
import datetime
import gpslib
from rtlsdr import RtlSdr
import multiprocessing as mp
import json
import pickle
import socket
import errno
from collections import deque
import signal
import linecache
import sys
import platform


# ======== global variables =======================================
# Only variables in gpsglob.py are intended for customization.

MEAS_RUNNING = True         # all processes are stopped after setting False
ASYNCIO_SLEEP_TIME = 1.e-4  # value must be > 0 in POSIX
SMP_TIME = np.int64(0)      # in units of 1/SAMPLE_RATE, ~0.5us; increased by
                            # number of samples NGPS with each stream; 
                            # timestamp for first value in data array
SEC_TIME = np.linspace(1,NGPS,NGPS,endpoint=True,
                       dtype=MY_FLOAT)/SAMPLE_RATE 
                            # array of sample times in s;
                            # used in demodDoppler for frequency sweep
SAT_ALL = list(range(2,33)) # currently active GPS satellites (PRN)
RESULT_LIST = []            # only used if SAVE_PICKLE = True 

# --- SDR-RTL -------------------------------------------------

SDR_CENTERFREQ = 1575.42e6  # in Hz, L1 band of GPS
SDR_GAIN = 50               # tuner gain in dB
SDR_BANDWIDTH = 0           # tuner bandwidth

# --- variables for data buffer ---------------------

MAXBUFSIZE = 16             # max. number of buffer slots for samples
BUFFER = deque([], maxlen=MAXBUFSIZE) 
                            # doubly ended queue; faster than list
NBUF = 0                    # currently used buffer slots
BUFSKIP = 0                 # counts number of skipped data streams
                            # due to buffer overflows 


# -------- use of Ctrl-C & Ctrl-Break ------------------------------

OS = platform.system()  # currently Windows and Linux are supported

def stopMeas(signal, frame):
    global MEAS_RUNNING
    MEAS_RUNNING = False
    if OS == 'Windows':
        print('CTRL-BREAK pressed')
    else:
        print('Ctrl-C pressed')

# start signal handler
if OS == 'Windows':
    signal.signal(signal.SIGBREAK, stopMeas)
else:
    signal.signal(signal.SIGINT, stopMeas)


# -------- Buffer ----------------------------------

def pushToBuffer(data):
    global BUFFER
    global NBUF
    global BUFSKIP

    if NBUF >= MAXBUFSIZE:
        BUFFER.clear()
        NBUF = 0
        BUFSKIP += MAXBUFSIZE
        
    BUFFER.append(data)
    NBUF += 1
    
    
def pullFromBuffer():
    global BUFFER
    global NBUF
    global BUFSKIP

    try:
        data = BUFFER.popleft()
        NBUF -= 1
        skip = BUFSKIP
        BUFSKIP = 0
    except IndexError:
        data = []
        skip = 0
    
    return data,skip


# ----------- Streaming real-time data ----------------        

async def streamLive():
    global MEAS_RUNNING

    sdr = RtlSdr()
    sdr.set_bias_tee(True)
    sdr.sample_rate = SAMPLE_RATE
    if SDR_FREQCORR != 0:
        sdr.freq_correction = SDR_FREQCORR    
    sdr.center_freq = SDR_CENTERFREQ   
    sdr.gain = SDR_GAIN 
    sdr.bandwidth = SDR_BANDWIDTH    

    sdrClosed = False
    measTimeout = False
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    end_time = start_time+MEAS_TIME
    try:
        async for samples in sdr.stream(num_samples_or_bytes=NGPS,
                                        format='samples'):
            pushToBuffer(samples)
            if loop.time()>end_time:                
                MEAS_RUNNING = False                    
                print('Timeout')
            if not MEAS_RUNNING:
                print('sdr to stop ..')
                await sdr.stop()
                print('sdr stopped')
                sdr.close()
                sdrClosed = True
                print('sdr closed')
    except:    
        print('Exception from sdr.stop')
    finally:
        MEAS_RUNNING = False
        if not sdrClosed:
            print('sdr to close ..')
            #sdr.close()        
            #print('sdr closed')
                    

# ----------- Streaming saved data (2 byte IQ) -----------


async def streamData(path,dataFile):
    global MEAS_RUNNING
    k = 0
    try:
        statusMsg = ''
        measTimeout = False
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        end_time = start_time+MEAS_TIME
        with open(path+dataFile,'rb') as f1:
            while k < START_STREAM:
                byteData = np.fromfile(f1,dtype=np.uint16,count=NGPS)
                k += 1
            while MEAS_RUNNING:
                if NBUF < 1:
                    byteData = np.fromfile(f1,dtype=np.uint16,count=NGPS)
                    if len(byteData)==NGPS:
                        im,re = np.divmod(byteData,256)
                        samples = np.asarray(
                                re+1j*im,dtype=np.complex64)/127.5 - (1+1j)
                        pushToBuffer(samples)
                    else:
                        MEAS_RUNNING = False
                        statusMsg = 'EOF'
                    if (loop.time()>end_time):                
                        statusMsg = 'Timeout'
                        MEAS_RUNNING = False
                await asyncio.sleep(ASYNCIO_SLEEP_TIME)
    except BaseException as err:
        printException()
        statusMsg = err
    finally:
        MEAS_RUNNING = False
        print(statusMsg)
             

# -------- JSON Encoder for numpy ----------------------

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
        
    
# ----------- Save Results --------------------------------           
            
def saveResults(path,saveDate):
    
    try:
        with open(f'{path}{saveDate}_gpsResult.pickle','wb') as file:
            pickle.dump(RESULT_LIST,file)
            
    except BaseException as err:
        print('Fehler: ',err)        


# ------- Find maximum in correlation (code phase) ---------

def findCodePhase(gpsCorr):
    mean = np.mean(gpsCorr)
    std = np.std(gpsCorr)
    delay = -1
    
    mx = np.argmax(gpsCorr)
    normMaxCorr = (gpsCorr[mx]-mean)/std
    if normMaxCorr > CORR_MIN:
        delay = mx
             
    return delay,normMaxCorr
    
            
# ------  demodulate doppler frequency, used for sweepFrequency ---------

def demodDoppler(data,dopplerFreq,dopplerPhase,N):                          
    factor = np.exp(-1j*(dopplerPhase+2*np.pi*dopplerFreq*SEC_TIME[:N]))
    dopplerPhase += 2*np.pi*dopplerFreq*SEC_TIME[N-1]
    return factor*data[:N], np.remainder(dopplerPhase,2*np.pi)


# -------- sweep frequency -------------
 

def sweepAllSats(data,freq,satLst,satFound,itSweep=2):   
    sweepReady = False
    avg = min(SWEEP_CORR_AVG,N_CYC)
    N = avg*CODE_SAMPLES
    
    phase = 0
    it = 0
    while freq < MAX_FREQ and it < itSweep:
        newData,_ = demodDoppler(data,freq,phase,N)    
        df = 0
        for i in range(avg):
            dfm = fft(newData[i*CODE_SAMPLES:(i+1)*CODE_SAMPLES]) 
            df += dfm
        fftData = df/avg

        removeLst = []
        for satNo in satLst: 
            corr = np.abs(ifft(fftData*np.conjugate(FFT_CACODE[satNo])))
            delay,normMaxCorr = findCodePhase(corr)
            if delay > -1:
                satFound.append((normMaxCorr,satNo,freq,delay))
                removeLst.append(satNo)

        for satNo in removeLst:
            satLst.remove(satNo)
        
        freq += STEP_FREQ
        if (freq >= MAX_FREQ):
            sweepReady = True
            freq -= MAX_FREQ - MIN_FREQ
            
        it += 1
        
    return sweepReady,freq,sorted(satFound,reverse=True)
   
          
# ------- Exception handling & Output -------------------


def printException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'\
          .format(filename, lineno,line.strip(), exc_obj))


def reportFoundSats(foundSats):
    print('Found Satellites:')
    for fs in foundSats:
        print(f'PRN {fs[1]:02d} Corr:{fs[0]:4.1f}  f={fs[2]:+7.1f}')
    print()


# ------- Multiprocessing functions --------------

def runProc(inQ,outQ):
    while True:
        msg = inQ.get()
        
        if msg[0]=='initPool':
            global WORKER_NO
    
            WORKER_NO = msg[1]
            process = mp.current_process()    
            name = process.name
            outQ.put((name,WORKER_NO))            
            
        elif msg[0]=='initInst':
            global SATPROC
            
            satNo,freq,delay = msg[1]
            SATPROC = gpslib.SatStream(satNo,freq,delay=delay,
                                       itSweep=IT_SWEEP,
                                       corrMin=CORR_MIN,
                                       corrAvg=CORR_AVG,
                                       sweepCorrAvg=SWEEP_CORR_AVG)
            outQ.put(satNo)            
            
        elif msg[0]=='delInst':           
            done = False
            if 'SATPROC' in globals():                
                del SATPROC
                done = True
            outQ.put(done)            
            
        elif msg[0]=='runInst':            
            data,smpTime = msg[1]
            swFq,frameData,coPh,cpQ = SATPROC.process(data,smpTime)
            satNo = SATPROC.SAT_NO
            outQ.put((swFq,satNo,frameData,coPh,cpQ))
        
        elif msg[0]=='done':
            break


def initMultiProcPool(poolNo):   
    pool = []
    for workerNo in range(poolNo):
        inQ = mp.Queue()
        outQ = mp.Queue()
        satProc=mp.Process(target=runProc, args=(inQ,outQ,),
                           daemon=True,name='worker%d' % (workerNo,))
        satProc.start()
        pool.append((inQ,outQ,satProc))
    
    pnoLst = []                     
    for workerNo,(inQ,_,_) in enumerate(pool):
        inQ.put(('initPool',workerNo))

    res = []
    for _,outQ,_ in pool:
        name,no = outQ.get()

    poolWorker = [0]*poolNo         # 0 = available; 
                                    # X = occupied with PRN X    
    return pool,poolNo,poolWorker


def closeMultiProcPool(pool):        
    for inQ,outQ,satProc in pool:
        inQ.put(('done',None))
        satProc.join()
        satProc.close()


def delPoolStreams(pool,poolNo,poolWorker,actSatSet,delSatSet):

    for satNo in delSatSet:
        wno = poolWorker.index(satNo)
        inQ,outQ,_ = pool[wno]
        inQ.put(('delInst',None))
        done = outQ.get()
        if done:
            poolWorker[wno] = 0     # 0 = available; 
                                    # X = occupied with PRN X            
    actSatSet = actSatSet - delSatSet            

    return poolWorker,actSatSet


def initPoolStreams(pool,poolNo,poolWorker,actSatSet,newSatSet,foundSats):

    if len(newSatSet) > 0:
        for wno,sno in enumerate(poolWorker):
            if sno == 0:
                newSat = newSatSet.pop()
                poolWorker[wno] = newSat
                inQ,outQ,_ = pool[wno]
                _,_,freq,delay = list(filter(lambda e:e[1]==newSat,
                                             foundSats))[0]
                inQ.put(('initInst',(newSat,freq,delay)))
                newSat = outQ.get()
                actSatSet.add(newSat)        
                if len(newSatSet) == 0:
                    break
                                   
    return poolWorker,actSatSet


def satCalc(actSatSet,pool,poolWorker,data,smpTime):
    outQLst=[]
    for sno in actSatSet:
        wno = poolWorker.index(sno)
        inQ,outQ,_ = pool[wno]
        inQ.put(('runInst',(data,smpTime)))
        outQLst.append(outQ)

    resLst=[]
    for outQ in outQLst:
        res = outQ.get()
        resLst.append(res)
        
    return resLst


# ------ select satellites -----------------------------------


def getNewSats(actSatSet,foundSats,cpQLst):
    # select currently active sats with sufficient correlation signal
    goodSatSet = set()
    for satNo in cpQLst:
        cpQ,cpL = cpQLst[satNo]
        if cpQ > 0 or cpL > 0:
            goodSatSet.add(satNo)
    # remove these from foundSats and select max number of new sats with 
    # largest correlation signal
    fs = list(filter(lambda item: item[1] not in goodSatSet,foundSats))
    maxfs = MAX_SAT - len(goodSatSet)        
    foundSatSet = goodSatSet | {item[1] for item in fs[:maxfs]}   

    commonSats = actSatSet & foundSatSet   # intersection 
    delSatSet = actSatSet - commonSats     # to be deleted
    newSatSet = foundSatSet - commonSats   # to be initialized                            
    
    return delSatSet,newSatSet
    

# ------- main - process data  -------------------

async def processData():   
    global MEAS_RUNNING
    global SMP_TIME
    global RESULT_LIST

    sweepAllFreq = True
    dopplerFreq = MIN_FREQ              

    sockUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
    sockUDP.bind(('', UDP_PORT_2))         # accept all
    sockUDP.setblocking(0)                 # no blocking

    pool,poolNo,poolWorker = initMultiProcPool(MAX_SAT)
    actSatSet = set()
    
    coPhLst,cpQLst = {},{}    
    skippedData = 0
    satLst = SAT_ALL.copy()
    foundSats = []
    msg = ''
    try:
        while MEAS_RUNNING:

            if NBUF >= 1:
                data,skip = pullFromBuffer()
                skippedData += skip*NGPS       # for report 
                SMP_TIME += (1 + skip)*NGPS    # timestamp for data[0]
                
                if sweepAllFreq:                 

                    sweepReady,dopplerFreq,foundSats \
                        = sweepAllSats(data,dopplerFreq,satLst,
                                       foundSats,itSweep=IT_SWEEP_ALL)
                    if sweepReady:
                        sweepAllFreq = False 
                        # sorted list of (corrMax,satNo,freq,delay)
                        reportFoundSats(foundSats)  
                        delSatSet,newSatSet \
                            = getNewSats(actSatSet,foundSats,cpQLst)
                        if len(delSatSet) > 0:
                            poolWorker,actSatSet \
                                = delPoolStreams(pool,poolNo,poolWorker,
                                                 actSatSet,delSatSet)
                        poolWorker,actSatSet \
                            = initPoolStreams(pool,poolNo,poolWorker,
                                              actSatSet,newSatSet,foundSats)
                                                
                else:                    
                    # start multiprocessing
                    resLst = satCalc(actSatSet,pool,poolWorker,data,SMP_TIME)
                                        
                    frameLst = []
                    streamNo = SMP_TIME//NGPS
                    for swFq,satNo,fLst,coPh,cpQ in resLst:
                        frameLst += fLst
                        cpQLst[satNo] = cpQ
                        if coPh >= 0:
                            if satNo in coPhLst:
                                coPhLst[satNo].append((streamNo,coPh))
                            else:
                                coPhLst[satNo] = [(streamNo,coPh)]
                                                                        
                    # all frames once per sec (also from sweeps); 
                    # cpPh every cycle (N_CYC ms)
                    if len(frameLst)>0:
                        res = pickle.dumps((skippedData,frameLst,coPhLst))
                        if SAVE_PICKLE:
                            RESULT_LIST.append(res)
                        if SEND_OVER_UDP:                            
                            try:
                                sockUDP.sendto(res, (UDP_IP, UDP_PORT))
                            except socket.error:
                                printException()
                        coPhLst = {}
                        skippedData = 0
                        
                try:
                    msg,addr = sockUDP.recvfrom(UDP_BUFSIZE_2)
                except ConnectionResetError:
                    pass
                except socket.error as err:
                    if err.args[0] != errno.EWOULDBLOCK:  
                        raise

                if (msg == b'SWEEP'):
                    msg = ''
                    sweepAllFreq = True
                    satLst = SAT_ALL.copy()
                    foundSats = []
                elif (msg==b'STOP'):
                      msg = ''
                      MEAS_RUNNING = False

            await asyncio.sleep(ASYNCIO_SLEEP_TIME)
            
    except BaseException as err:
        printException()
    finally:            
        MEAS_RUNNING = False
        if SAVE_PICKLE:
            saveResults(DATA_PATH,SAVE_DATE)
        sockUDP.close()
        closeMultiProcPool(pool)
        print('Pool closed')        
    
              
# --------------- Main ------------------        

async def main():
    start = time.time()
    loop = asyncio.get_event_loop()
    task2 = loop.create_task(processData())
    if LIVE_MEAS:
        task1 = loop.create_task(streamLive())
    else:
        task1 = loop.create_task(streamData(DATA_PATH,BIN_DATA))

    await task2
    print('Task2 finished')
    await asyncio.wait_for(task1,1)
    print('Task1 finished')
    
    print(time.time()-start)
 
 
if __name__ == "__main__":
            
    mp.set_start_method('spawn')   # default in POSIX is 'fork' 

    FFT_CACODE = [0,0]             # position in list is PRN; 
                                   # first PRN in SAT_ALL is 2
    for sat in SAT_ALL:            # FFTs for correlation in sweepFreqeuncy
        FFT_CACODE.append(fft(gpslib.GPSCacode(sat)))
        
    asyncio.run(main()) 

