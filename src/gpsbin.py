# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

import asyncio
import numpy as np
from rtlsdr import RtlSdr


# ------- data buffer -----------------

def pushToBuffer(data):
    global MEAS_RUNNING
    global buffer
    global nbuf
    buffer += data
    nbuf += len(data)
    if nbuf > maxBufSize:
        MEAS_RUNNING = False
    
def pullFromBuffer(n):
    global buffer
    global nbuf
    if n > nbuf:
        n = nbuf        
    y = buffer[:n]
    buffer = buffer[n:]
    nbuf -= n
    return y,n


# -------- process data -----------------            
            
async def processData():   
    global MEAS_RUNNING
    k = 0
    noStream = 0
    noMin = 0
    print(noMin,end=' ')
    try:
        f1 = open(FILENAME,'wb')
        while MEAS_RUNNING or nbuf>0:
            k += 1
            if nbuf >= N or not MEAS_RUNNING:
                data,n = pullFromBuffer(N)            
                data = np.asarray(data,dtype = np.uint8) # 2 bytes per sample
                data.tofile(f1)                
                k = 0
                noStream += 1
                print('*',end='',flush=True)
                if noStream % 12 == 0:
                    print(' ',end='',flush=True)
                if noStream % 60 == 0:
                    noMin += 1
                    print()
                    print(noMin // 4,end=' ',flush=True)
            await asyncio.sleep(SLEEP_TIME)
    except BaseException as err:
        print('Fehler: ',err)
    finally:
        f1.close()        
        MEAS_RUNNING = False


# ----------- Streaming ----------------        


async def streaming():
    global MEAS_RUNNING
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    end_time = start_time+meas_time
    try:
        async for samples in sdr.stream(num_samples_or_bytes=N, format='bytes'):
            pushToBuffer(samples)
            if (loop.time()>end_time) or not MEAS_RUNNING:
                MEAS_RUNNING = False
                await sdr.stop()
    except:    
        print('\n\nSDR-Error')
    finally:
        sdr.close()      
        
# -------- parameter -------------------        

SLEEP_TIME = 1.0E-4      # in seconds
MEAS_RUNNING = True      # global: if False, all processes are stopped

buffer = []              # global data buffer for reading samples from rtl-sdr
nbuf = 0                 # current data size, see maxBufSize below 

# configuration SDR
sample_rate = 2.048e6   
freq_correction = -2     # in ppm
center_freq = 1575.42e6   
gain = 50.0              # tuner gain in dB
bandwidth = 0            # tuner bandwidth

N = 1024000
maxBufSize = 4*N
meas_time = 2*60  # in seconds, 1 min is equivalent to 245.76 MB 

FILENAME = 'data_101024_1.bin'

# --------------- Main ------------------        

sdr = RtlSdr()
sdr.set_bias_tee(True)
sdr.sample_rate = sample_rate
sdr.freq_correction = freq_correction    
sdr.center_freq = center_freq   
sdr.gain = gain 
sdr.bandwidth = bandwidth    


async def main():
    print('A block of 12 asterisks represent 3 s, a row 15 s.')
    loop = asyncio.get_event_loop()
    task2 = loop.create_task(processData())
    task1 = loop.create_task(streaming())
    
    await task2
    print('Task2 finished')
    await asyncio.wait_for(task1,1)
    #await task1
    print('Task1 finished')
 
 
if __name__ == "__main__":
        
    asyncio.run(main()) 
