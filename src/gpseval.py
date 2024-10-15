# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

from gpsglob import *
import matplotlib as mpl
mpl.use('qtagg')
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import datetime
import gpslib
import gpsui
import json
import socket
import errno
import pickle
import gpxpy
import time


# ======== global variables ========================================
# Only variables in gpsglob.py are intended for customization.

MEAS_RUNNING = True        # if False, all processes are stopped

# --- global lists and dictionairies ----------

ORB_LIST = {}              # save instances of SatOrb()
FRAME_LIST = []            # list of subframes
COPH_LIST = {}             # original code phases from measurement
SAT_LOG = {}               # log errors and applied corrections
SAT_RES = {}               # range estimates for satellites;
                           # list of (tow,streamNo,rangeEst,measDelay)
SATRES_LIST = []           # list of positions from satellites; tuples contain
                           # (sat_No,tow,x,y,z,st+d_st,weekNum,cycNo,cophStd)
POS_LIST = []              # list of calculated ECEF positions (with outliers)
OUTLIER_LIST = []          # list of outliers not used for positioning
FAIL_LIST = []             # errors in fixing position;
                           # list of (tow,streamNo,cause);
                           # cause is either 'MAX_RESIDUAL' or 'EXCEPTION'
EPHEMERIDES = {}           # orbit parameter of satellites


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


# ------ Save and load ephemerides of satellites -------

def saveEphemerides(path,ephemfile):
    global ORB_LIST
    date = datetime.datetime.now(datetime.UTC)

    ephem = {}
    # for human readability
    ephem['UTC-Time'] = date.strftime('%Y-%m-%d %H:%M:%S')
    ephem['POSIX-Time'] = int(date.timestamp())

    for satNo in ORB_LIST:
        satOrbit = ORB_LIST[satNo]
        if satOrbit.data.ephemOk:
            ephem[satNo] = satOrbit.data.ephemData
    try:
        with open(path+ephemfile,'w') as file:
            json.dump(ephem,file,indent=2,cls=MyEncoder)
    except:
        printException() 


def loadEphemerides(path,ephemfile):
    currentTime = datetime.datetime.now(datetime.UTC)
    ephem = {}
    ephemTime = 0
    try:
        with open(path+ephemfile,'r') as file:
            ephDict = json.load(file)

        for key in ephDict:
            if key == 'POSIX-Time':
                ephemTime = ephDict[key]
            elif key == 'UTC-Time':
                pass
            else:
                # keys are saved as strings in JSON
                ephem[int(key)] = ephDict[key]

        # Ephemeris is typically valid for ~2 h
        if currentTime.timestamp()-ephemTime > 2*3600:
            ephem = {}
    except FileNotFoundError:
        pass
    except:
        printException()
        ephem = {}

    return ephem


# ----------- Load measured data from file ---- -------------------------

def loadMeasData(path,datafile):
    try:
        with open(path+datafile,'rb') as file:
            measData = pickle.load(file)
    except:
        measData = None
        printException()
        
    return measData
        
    
# ----------- Save Positions, Frames and CodePhases ---------------------

def saveResults(path,saveDate):
    try:
        with open(f'{path}{saveDate}_gpsFrames.json','w') as file:
            json.dump(FRAME_LIST,file,indent=2,cls=MyEncoder)
                        
        with open(f'{path}{saveDate}_gpsSatRes.json','w') as file:
            json.dump(SATRES_LIST,file,indent=2,cls=MyEncoder)
            
        with open(f'{path}{saveDate}_gpsPos.json','w') as file:
            json.dump(POS_LIST,file,indent=2,cls=MyEncoder)
            
        with open(f'{path}{saveDate}_gpsCP.json','w') as file:
            json.dump(COPH_LIST,file,indent=2,cls=MyEncoder)
    except:
        printException()
                            
            
def saveGeoTrack(geoTrack,path,saveDate):
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for lon,lat,elev in geoTrack:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(
                                  lon,lat, elevation=elev))
    try:
        with open(f'{path}{saveDate}_gpsTrack.gpx','w') as file:
            file.write(gpx.to_xml())

    except:
        printException() 


# ------- Exception handling  -------------------

import linecache
import sys

EXC = ''

def printException():
    global EXC
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    EXC = ('EXCEPTION IN ({}, LINE {} "{}"): {}'\
           .format(filename, lineno, line.strip(), exc_obj))
    print(EXC)
    
        
# ------- evaluate ---------------------------------

def getSatOrbit(satNo):
    global ORB_LIST
    global EPHEMERIDES
    global SAT_LOG
    
    if satNo in ORB_LIST:
        satOrbit = ORB_LIST[satNo]
    else:
        SAT_LOG[satNo] = ['LOG for Sat %d' % (satNo)]
        ephem = EPHEMERIDES[satNo] if satNo in EPHEMERIDES else None
        satOrbit = gpslib.SatOrbit(satNo,eph=ephem)
        ORB_LIST[satNo] = satOrbit

    return satOrbit


def evalData(frameLst,cpLst,errLst,swpLst):
    satResLst = []
    gpsTime = 0
    # A frame list is generated once per second and then contains a subframe
    # for each sat. For sweeps frameLst is generated every N_CYC millisec 
    # and does NOT contain all sats.
    for sf in frameLst:
        satNo = sf['SAT']
        satOrbit = getSatOrbit(satNo)
        sf['EPH'] = 'Ok' if satOrbit.data.ephemOk else ''
        if 'SWP' in sf and sf['SWP']:
            swpLst[satNo] = 'sweep'
            
        if 'ID' in sf:
            satOrbit.readFrame(sf)

    actSats = set()
    # the currently active sats in cpLst and frameLst are not identical
    for satNo in cpLst:
        satOrbit = getSatOrbit(satNo)
        status = satOrbit.status

        if status > satOrbit.notReady:
            errLst[satNo] = satOrbit.errmsg

        result = satOrbit.evalCodePhase(cpLst[satNo],relCorr=True) 
        # list of tuples (sat_No,tow,x,y,z,st+d_st,weekNum,cycNo,cophStd)
        if len(result) > 0:
            satResLst += result
            actSats.add(satNo)
            if gpsTime == 0:
                tow,weeknum,cycNo = result[0][1],result[0][6],result[0][7]
                gpsTime = gpslib.gpsTime(tow,weeknum) \
                          + datetime.timedelta(seconds=cycNo*N_CYC/1000)
                                    
    return satResLst,errLst,swpLst,actSats,gpsTime
  

def ecefPositions(satResLst,ecefPosStat):
# ----- Input ----------------------------------------
# satResLst is list of
#   sat_No : PRN of satellite
#   tow    : time of week as time reference (preamble)
#   x,y,z  : satellite position (ECEF)
#   st+d_st: corrected sample time (local time)
#   weekNum: week number
#   cycNo  : time lag to tow; cycNo*N_CYC in ms (+offset)
#   cophStd: standard deviation of code phase
# ----- Output ----------------------------------------
# SAT_RES[satNo] is list of
#   tow    : time of week
#   cycNo  : time lag to tow
#   range  : distance satellite to receiver
#   delay  : measured propagation time of signal (in m)
# recPosList is list of
#   gpsTime: time of measurement
#   x,y,z  : recorder position (ECEF)
# ------------------------------------------------------
    minSat = 3 if CONF_HEIGHT else 4
    calcSat = max(MIN_SAT,minSat)
    recPosLst,lsfFailLst,satPosLst = [],[],[]
    ptow,r = 0,0
    locStart = [0,0,0,0]
    if ecefPosStat is not None:
        locStart[1:] = ecefPosStat[0]         # previous fix
    
    # sort result list in order tow, cycNo, satNo
    satResLst.sort(key = lambda e: (e[1],e[7],e[0]))
    while r < len(satResLst):
        # filter entries with equal (tow,cycNo)
        bLst=[satResLst[r]]
        tow,cycNo = satResLst[r][1],satResLst[r][7]
        r += 1
        while r < len(satResLst) and \
        (satResLst[r][1],satResLst[r][7]) == (tow,cycNo):
            bLst.append(satResLst[r])
            r += 1
        nSat = len(bLst)
        if nSat >= calcSat:
            # with a sufficient number of sats fix the receiver position
            satCoord = np.zeros((3,nSat))
            timeDel = np.zeros(nSat)
            timeStd = np.zeros(nSat)
            for i in range(nSat):
                weeknum = bLst[i][6]           # weeknum
                satCoord[:,i]=bLst[i][2:5]     # (x,y,z)
                timeDel[i] = bLst[i][5]        # sample time
                timeStd[i] = bLst[i][8]        # cophStd
                if tow != ptow:
                    # for sat position in plot; one entry per subframe
                    satPosLst.append((bLst[i][0],bLst[i][2:5]))
            ptow = tow
            stdDev = timeStd if LSF_WEIGHT else None
            try:
                # output recPosition is (t0,x,y,z)
                recPosition,residuals,rangeEst,measDelay \
                    = gpslib.leastSquaresPos(minSat,satCoord,timeDel,
                                             maxResidual=MAX_RESIDUAL,
                                             maxIt=LSF_MAX_IT,recPos=locStart,
                                             height=HEIGHT,hDev=HEIGHT_DEV,
                                             stdDev=stdDev)
            except:
                lsfFailLst.append((tow,cycNo,'EXCEPTION'))
            else:
                if residuals[-1] <= MAX_RESIDUAL:
                   # fit was succesful; save in list SAT_RES
                    gpsTime = gpslib.gpsTime(tow,weeknum) \
                              + datetime.timedelta(seconds=cycNo*N_CYC/1000)
                    recPosition[0] = gpsTime.timestamp()  # POSIX-Time in sec
                    recPosLst.append(recPosition)
                    for i in range(nSat):
                        satNo = bLst[i][0]
                        if satNo not in SAT_RES:
                            SAT_RES[satNo] = []
                        SAT_RES[satNo].append((tow,cycNo,rangeEst[i],
                                               measDelay[i]))
                else:
                    # accuracy of fit not sufficient; save in fail list
                    lsfFailLst.append((tow,cycNo,'MAX_RESIDUAL'))
                                            
    return satPosLst,recPosLst,lsfFailLst
    

# ------- calc mean position with 1s period ------------
       
def meanSecPosition(recPosLst):
    r = np.asarray([item[1:] for item in recPosLst])
    meanSecPos = np.mean(r,axis=0)              # (xmean,ymean,zmean)
        
    return meanSecPos

    
# ------- statistics of positions (ECEF coordinates) ------


def ecefStatistics(allStat,allPosLst,recPosLst,lastPosTime):
    global OUTLIER_LIST
    
    r = [item[1:] for item in recPosLst]        # max number is 1024/N_CYC
    newPosTime = recPosLst[0][0]                # POSIX time in sec
    if lastPosTime is None:
        lastPosTime = newPosTime

    if allStat is not None:
        allMean = allStat[0]                    # previous average position
    else:
        allMean = np.mean(r,axis=0)             # (xmean,ymean,zmean)
    
    # include time gaps between fixes to identify outliers
    minOutDist = MIN_OUT_DIST + (newPosTime-lastPosTime) * MAX_SPEED
    # distance to mean value for each fixed position
    drLst = np.linalg.norm(np.asarray(r)-allMean,axis=1)
    # position is regarded as outlier if distance to mean > minOutDist
    outLst = [i for i,dr in enumerate(drLst) if dr > minOutDist]
    # remove outliers from result list
    for i in reversed(outLst):
        OUTLIER_LIST.append(recPosLst[i])
        del r[i]
        del recPosLst[i]

    # calculate mean position and standard deviation
    if r != []:
        allPosLst += r
        lastPosTime = recPosLst[-1][0]
        noPosAvg = POS_AVG_IN_SEC * (1024 // N_CYC)
        avgPos = allPosLst[-noPosAvg:] if noPosAvg > 0 else allPosLst
        allMean = np.mean(avgPos,axis=0)
        allDev = np.std(avgPos,axis=0)
        allN = len(avgPos)
        allOut = len(OUTLIER_LIST)
        allStat = (allMean,allDev,allN,allOut)
        
    return allStat,allPosLst,recPosLst,lastPosTime


# ------- evaluate code phase ----------------------------

# Overflow and phase error detection of code phases
# max code phase drift is 6.6/sec; 
# typ. sd is 0.05-0.10; drift over 32 ms < 6.6/32 = 0.206
def cpOflCorrection(satNo,cplst,errStream):
    global SAT_LOG

    diffTol = 200                # tolerance for neighboured code phase values
    maxDiffNo = N_CYC // 4       # allowed gap between stream numbers for code
                                 # phases of succeeding lists
    cpl = cplst.copy()
    ovfl = 0
    pno,pcp = cpl[0]
    # first, correct overflow within stream
    for i in range(1,len(cpl)):
        no,cp = cpl[i]
        cp += ovfl*CODE_SAMPLES
        diff = pcp - cp
        if np.isclose(abs(diff),CODE_SAMPLES,rtol=1E-5,atol=diffTol):
            cp += np.sign(diff)*CODE_SAMPLES
            ovfl += np.sign(diff)
        if abs(cp-pcp) > (1+(no-pno-1)*0.2):
            SAT_LOG[satNo].append('%d [%d]: cpMaxDev(1) error:%1.2f - '\
                           'possible phase error' % (no,no-pno,abs(cp-pcp)))
            # phase error happens somewhere in the range (pno,no]
            for j in range(no-pno):
                errStream[no-j] = errStream[no-j]+1 if no-j in errStream else 1
        cpl[i] = (no,cp)
        pno,pcp = no,cp
    # now find phase errors in comparison to previous code phase list
    if len(cpl) > 0 and satNo in COPH_LIST and len(COPH_LIST[satNo]) > 0:
        no,cp = cpl[0]                       # first entry in new list
        pno,pcp = COPH_LIST[satNo][-1]       # last entry of previous list
        if no - pno <= maxDiffNo:
            diff = pcp - cp
            if np.isclose(abs(diff),CODE_SAMPLES,rtol=1E-5,atol=diffTol):
                cp += np.sign(diff)*CODE_SAMPLES
            if abs(cp-pcp) > (1+(no-pno-1)*0.2):
                SAT_LOG[satNo].append('%d [%d]: cpMaxDev(2) error:%1.2f - '\
                            'possible phase error' % (no,no-pno,abs(cp-pcp)))
                for j in range(no-pno):
                    errStream[no-j] \
                        = errStream[no-j]+1 if no-j in errStream else 1
    
    return cpl,errStream


# Rarely several datastreams get lost (e.g. overwritten in buffer) and as a
# result the time references are no longer valid. It is important to capture
# these errors before getting an endless bunch of failures when calculating
# receiver positions. A straightforward way to do that is to compare the code
# phases of all satellites and find correlated jumps in the curves. Therefore,
# phase error detection has to be done before the code phases are passed to
# SatOrbit() instances. In the case of small time gaps between succeeding code
# phases, phase error detection is quite simple (see cpOflCorrection).
# For larger gaps (between lists) error detection can be omitted, since a large
# gap for more than one satellite is rare and correlated errors of few
# satellites is sufficient (currently 3). Besides error detection, code phases
# must also be corrected regarding overflow (jumps from 0 to 2047 or from 2047
# to 0). This is done here only for overflow WITHIN a code phase list, so the
# first value in the list is always within [0,2047]. The overflow correction
# BETWEEN lists is done later in SatOrbit(), because the offset for overflow
# correction is reset with a new time reference.
def prepCodePhase(coPhLst,noPhaseErr):
    global SAT_LOG

    minSatErr = 3
    minEntries = N_CYC // 4
    cpl,errStream = {},{}
    for satNo in coPhLst:
        if len(coPhLst[satNo]) >= minEntries:
            cpl[satNo],errStream = cpOflCorrection(satNo,coPhLst[satNo],
                                                   errStream)
    if len(errStream) > 0 and max(errStream.values()) >= minSatErr:
        # at least three sats have an error at the same time
        keySNO = max(errStream, key=errStream.get)
        satErr = max(errStream.values())
        for satNo in SAT_LOG:
            SAT_LOG[satNo].append('%d: phase error(%d) - new time ref'\
                                  % (keySNO,satErr))
            # notifies sats to discard all frames before stream keySNO
            cpl[satNo] = [(keySNO,None)]
        noPhaseErr += 1
                    
    return cpl,noPhaseErr
        
    
# ------- main - process data  ----------------------------

def processData():
    global MEAS_RUNNING
    global FRAME_LIST
    global SATRES_LIST
    global POS_LIST
    global COPH_LIST
    global FAIL_LIST
    global OUTLIER_LIST
    global LOAD_PICKLE
    global CONF_HEIGHT
    global HEIGHT

    # initalize user interface
    gpsUI = gpsui.GpsUI(FIG_SIZE,MIN_OUT_DIST,plotScale=PLOT_SCALE,\
                        posSize=POS_SIZE,height=HEIGHT,confHeight=CONF_HEIGHT)

    resetStats = True
    dataIdx = 0
    addr = None
    sockUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sockUDP.bind((UDP_ALL, UDP_PORT))  # binding is for receiving messages
    except:
        printException()
        plt.close()
        MEAS_RUNNING = False
        
    sockUDP.setblocking(0)

    allSkipData,noPhaseErr,nof = 0,0,0
    geoSecTrack = []                        # gpx is not reset with RESET_STATS
    gstStart = 0                            # first location to show on map
    try:
        while MEAS_RUNNING:
            
            if resetStats:
                ecefAllPos,OUTLIER_LIST,FAIL_LIST = [],[],[]
                errLst,swpLst = {},{}
                lastPosTime,ecefPosStat = None,None
                firstMeanPos,geoFirstPos = None,None
                allSkipData = 0
                meanSec = (0,0)
                locTrack,meanSecTrack = [(0,0)],[(0,0)]
                gpsUI.printMeanSecTrack(meanSecTrack)
                gpsUI.printTrack(locTrack)
                gstStart = len(geoSecTrack)
                resetStats = False

            events,confHeight,txtHeight = gpsUI.getEvents()
            trigStop,trigClose,trigSweep,trigShowMap,resetStats,zoom = events
            CONF_HEIGHT = confHeight
            HEIGHT = int(txtHeight)
            loc = None
            try:
                if LOAD_PICKLE:
                    dataPck = MEAS_DATA[dataIdx]
                    dataIdx += 1
                else:
                    dataPck, addr = sockUDP.recvfrom(UDP_BUFSIZE_1)
                    
            except IndexError:
                # end of MEAS_DATA is reached, switch to UDP
                LOAD_PICKLE = False
            except socket.error as e:
                # EWOULDBLOCK is raised if no data are received (non-blocking)
                if e.args[0] != errno.EWOULDBLOCK:
                    raise
            else:
                # typically once per second; frameLst != []
                skipData,frameLst,coPhLst = pickle.loads(dataPck)
                allSkipData += skipData
                
                cpLst,noPhaseErr = prepCodePhase(coPhLst,noPhaseErr)
                satResLst,errLst,swpLst,actSats,gpsTime \
                    = evalData(frameLst,cpLst,errLst,swpLst)
                satPosLst,recPosLst,failLst \
                    = ecefPositions(satResLst,ecefPosStat)
                POS_LIST += recPosLst
                FAIL_LIST += failLst

                gpsUI.printTime(gpsTime)
                
                nof += 1
                idFrames  = list(filter(lambda item: 'ID' in item,frameLst))
                if len(idFrames) > 0 or nof % 6 == 0:    # once in 6 sec
                    nof = 0                              # for synchronisation
                    gpsUI.printSatData(frameLst,actSats,errLst,swpLst)
                    errLst,swpLst = {},{}
                
                if len(recPosLst) > 0:
                    ecefPosStat,ecefAllPos,recPosLst,lastPosTime \
                        = ecefStatistics(ecefPosStat,ecefAllPos,
                                         recPosLst,lastPosTime)
                    
                if len(recPosLst) > 0:
                    if firstMeanPos is None:
                        firstMeanPos = ecefPosStat[0]
                        geoFirstPos = gpslib.ecefToGeo(firstMeanPos)
                        
                    geoMeanPos = gpslib.ecefToGeo(ecefPosStat[0])
                    gpsUI.printStat(ecefPosStat,geoMeanPos)
                    gpsUI.printSatPos(satPosLst,firstMeanPos)

                    for recPos in recPosLst:
                        geoPos = gpslib.ecefToGeo(recPos[1:])
                        loc = gpslib.locDistFromLatLon(geoFirstPos,geoPos)
                        locTrack.append(loc)
                    gpsUI.printTrack(locTrack)

                    meanSecPos = meanSecPosition(recPosLst)
                    geoSecPos = gpslib.ecefToGeo(meanSecPos)
                    geoSecTrack.append(geoSecPos)
                    meanSec = gpslib.locDistFromLatLon(geoFirstPos,geoSecPos)
                    meanSecTrack.append(meanSec)
                    gpsUI.printMeanSecTrack(meanSecTrack)

                gpsUI.printErrors(allSkipData//NGPS,len(OUTLIER_LIST),
                                  len(FAIL_LIST),noPhaseErr)

                for satNo in coPhLst:
                        COPH_LIST[satNo] = COPH_LIST[satNo]+coPhLst[satNo] \
                                     if satNo in COPH_LIST else coPhLst[satNo]
                FRAME_LIST += frameLst
                SATRES_LIST += satResLst
                                                        
            gpsUI.setPrintScale(meanSec,zoom)
            if trigShowMap:
                gpsUI.showMap(geoSecTrack[gstStart:])
            if trigSweep and addr is not None:
                sockUDP.sendto(b'SWEEP',(addr[0],UDP_PORT_2))
            if trigStop:
                LOAD_PICKLE = False
                if addr is not None:
                    sockUDP.sendto(b'STOP',(addr[0],UDP_PORT_2))
            if trigClose:
                if addr is not None:
                    sockUDP.sendto(b'STOP',(addr[0],UDP_PORT_2))
                MEAS_RUNNING = False
                
            gpsUI.updateCanvas()
            time.sleep(UDP_WAIT_TIME)
            
    except BaseException:
        printException()
    finally:
        MEAS_RUNNING = False
        sockUDP.close()
        
    try:
        if SAVE_EVAL_RES:
            saveResults(DATA_PATH,SAVE_DATE)
        if SAVE_EPHEM:
            saveEphemerides(DATA_PATH,EPHEM_FILE)
        if SAVE_TRACK:
            saveGeoTrack(geoSecTrack,DATA_PATH,SAVE_DATE)
    except BaseException as err:
        printException()
    

# ------- main -------------------

if __name__=='__main__':
    if LOAD_PICKLE:
        MEAS_DATA = loadMeasData(DATA_PATH,PICKLE_FILE)
        UDP_WAIT_TIME = 0
        
    if LOAD_EPHEM:
        EPHEMERIDES = loadEphemerides(DATA_PATH,EPHEM_FILE)

    processData()
