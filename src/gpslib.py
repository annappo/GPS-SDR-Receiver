# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

from gpsglob import \
    MY_FLOAT,MY_COMPLEX,\
    MIN_FREQ,MAX_FREQ,STEP_FREQ,\
    SAMPLE_RATE,N_CYC,CODE_SAMPLES,NGPS
import numpy as np
from scipy.fft import fft, ifft
import datetime
import math
from cacodes import cacodes

WEEK_IN_SEC = 604800
GPS_C  = 2.99792458e8           # as defined for GPS
GPS_PI = 3.1415926535898        # ditto
OMEGA_EARTH = 7.292115147e-5    # rad/sec

ROLLOVER = 2                    # for GPS time, next rollover is on Sun,
                                # 21.11.2038 00:00  + leapseconds for UTC
LEAPSEC = 18                    # in s; difference of GPS and UTC time


# -- Sets of quantities in subframes or from calculations -----------------

# for information only, not used in code
subFrame1 = {'ID','tow','weekNum','satAcc','satHealth','Tgd','IODC','Toc',
             'af2','af1','af0','ST'}
subFrame2 = {'ID','tow','Crs','deltaN','M0','Cuc','IODE2','e','Cus','sqrtA',
             'Toe','ST'}
subFrame3 = {'ID','tow','Cic','omegaBig','Cis','i0','IODE3','Crc','omegaSmall',
             'omegaDot','IDOT','ST'}
subFrame4 = {'ID','tow','ST'}
subFrame5 = {'ID','tow','ST'}

# used sets in Class SatData()
ephemSF1  = {'weekNum','Tgd','Toc','af2','af1','af0','IODC','satAcc'}
ephemSF2  = {'Crs','deltaN','M0','Cuc','e','Cus','sqrtA','Toe','IODE2'}
ephemSF3  = {'Cic','omegaBig','Cis','i0','Crc','omegaSmall','omegaDot','IDOT',
             'IODE3'}

# Calculated quantities in class SatPos() - not used in code
orbCalc  = {'Omega_k','M_k','E_k','nu_k','Phi_k','d_ik','i_k','d_uk','u_k',
            'd_rk','r_k','op_xk','op_yk'}
timeCalc = {'tsv','dtsv','gpst','tk','dtr'}

# Needed representation of quantities  -  may be helpful for saving in text
# files (not yet used)
boolVal  = {'SWP'}
intVal   = {'SAT','ID','tow','ST','weekNum','satAcc','satHealth','IODC',
            'IODE2','IODE3','Toc','Toe'}
floatVal = {'height','lat','lon','AMP','CRM','FRQ','Tgd','af2','af1','af0',
            'Crs','deltaN','M0','Cuc','e','Cus','sqrtA','Cic','omegaBig',
            'Cis','i0','Crc','omegaSmall','omegaDot','IDOT','Omega_k','M_k',
            'E_k','nu_k','Phi_k','d_ik','i_k','d_uk','u_k','d_rk','r_k',
            'op_xk','op_yk','tsv','dtsv','gpst','tk','dtr'}
string   = {'EPH'}


# ------ doubled GPS cacodes -------------------

def doubledCacode(satNo):
    code = cacodes[satNo]
    y = []
    for i in range(len(code)):
        y += [code[i],code[i]]
    y = np.asarray(y,dtype=MY_FLOAT)
    x = np.arange(len(y),dtype=MY_FLOAT)
    return x, y

# ------ Interpolation of cacode to 2048 points (CODE_SAMPLES) ----

def GPSCacode(satNo):
    x,y = doubledCacode(satNo)
    xp = np.linspace(x[0],x[-1],CODE_SAMPLES,endpoint=True,dtype=MY_FLOAT)
    yp = np.interp(xp,x,y)
    return yp

# ------------------------------------------------

def GPSCacodeRep(satNo,NCopies,delay):
    y0 = GPSCacode(satNo)
    y1 = y0
    for _ in range(NCopies-1):
        y1 = np.append(y1,y0)
    y1 = np.roll(y1,delay)
    return y1


# ----------- Class Subframe -----------------------------------------------
# Input for the function Extract() is a stream of bits (int8, 1 and 0)
# representing a subframe of length 300 beginning with the preamble (normal or
# inverted). Output is a status word. If valid the decoded subframe parameters
# are saved in corresponding variables of the object instance.

class Subframe():
    errlst = ['no error',
              'wrong length of subframe',
              'preamble error',
              'parity error',
              'no valid ID',
              'empty']
    noErr = 0
    lengthErr = 1
    preambleErr = 2
    parityErr = 3
    idErr = 4
    noData = 5
    preamble = np.array([1,0,0,0,1,0,1,1],dtype=np.int8)

    def __init__(self):
        self._words = np.zeros((10,30),dtype=np.int8)
        self._status = self.noData
        # all subframes
        self._ID = 0             # subframe ID 1 to 5
        self._tow = 0            # time of week count

        # subframe 1
        self._weekNum = 0        # week number
        self._satAcc = 0         # satellite accuracy
        self._satHealth = 0      # satellite health
        self._Tgd = 0            # in s, correction term for group delay
                                 # differential
        self._IODC = 0           # Issue of Data, for detection of changes in
                                 # parameters, see IODE2, IODE3
        self._Toc = 0            # clock data reference time; for calculation
                                 # of code phase offset
        self._af2 = 0            # polynomial coefficient; do.
        self._af1 = 0            # polynomial coefficient; do.
        self._af0 = 0            # polynomial coefficient; do.

        # subframe 2
        self._IODE2 = 0          # Issue of Data (ephemeris); see IODC, IODE3
        self._Crs = 0            # for correction of orbit radius
        self._deltaN = 0         # mean motion difference from computed value
        self._M0 = 0             # mean anomaly at reference time
        self._Cuc = 0            # for correction of argument of latitude
        self._Cus = 0            # for correction of argument of latitude
        self._e = 0              # eccentricity
        self._sqrtA = 0          # square root of the semi-major axis
        self._Toe = 0            # reference time ephemeris (or epoch time)

        # subframe 3
        self._Cic = 0            # for correction of angle of inclination
        self._Cis = 0            # for correction of angle of inclination
        self._omegaBig = 0       # longitude of ascending node of orbit plane
                                 # at weekly epoch
        self._i0 = 0             # inclination angle at reference time
        self._Crc = 0            # for correction of orbit radius
        self._omegaSmall = 0     # argument of perigree ('erdnächster Ort')
        self._omegaDot = 0       # rate of right ascension
        self._IDOT = 0           # rate of inclination angle
        self._IODE3 = 0          # Issue of Data (ephemeris), see IODE2, IODC

    @property
    def word(self,value):
        return self._words[value,:]

    @property
    def ID(self):
        return self._ID

    @property
    def tow(self):
        return self._tow

    @property
    def status(self):
        return self._status

    @property
    def errmsg(self):
        return self.errlst[self._status]

    @property
    def weekNum(self):
        return self._weekNum

    @property
    def satAcc(self):
        return self._satAcc

    @property
    def satHealth(self):
        return self._satHealth

    @property
    def Tgd(self):
        return self._Tgd

    @property
    def IODC(self):
        return self._IODC

    @property
    def Toc(self):
        return self._Toc

    @property
    def af2(self):
        return self._af2

    @property
    def af1(self):
        return self._af1

    @property
    def af0(self):
        return self._af0
    @property
    def IODE2(self):
        return self._IODE2

    @property
    def Crs(self):
        return self._Crs

    @property
    def deltaN(self):
        return self._deltaN

    @property
    def M0(self):
        return self._M0

    @property
    def Cuc(self):
        return self._Cuc

    @property
    def e(self):
        return self._e

    @property
    def Cus(self):
        return self._Cus

    @property
    def sqrtA(self):
        return self._sqrtA

    @property
    def Toe(self):
        return self._Toe

    @property
    def Cic(self):
        return self._Cic

    @property
    def omegaBig(self):
        return self._omegaBig

    @property
    def Cis(self):
        return self._Cis

    @property
    def i0(self):
        return self._i0

    @property
    def Crc(self):
        return self._Crc

    @property
    def omegaSmall(self):
        return self._omegaSmall

    @property
    def omegaDot(self):
        return self._omegaDot

    @property
    def IDOT(self):
        return self._IDOT

    @property
    def IODE3(self):
        return self._IODE3

    def Extract(self,subframeData):
        if len(subframeData) != 300:
            self._status = self.lengthErr
        else:
            data = np.copy(subframeData)
            preOk = (data[:8] == self.preamble).all()
            if not preOk:
                data = 1 - data
                preOk = (data[:8] == self.preamble).all()
            if not preOk:
                self._status = self.preambleErr
            else:
                self._words = np.reshape(data,(10,30))
                if self.CheckParity() > 0:
                    self._status = self.parityErr
                else:
                    self._tow = self.BinToInt(self._words[1,:17])
                    self._ID = self.BinToInt(self._words[1,19:22])
                    if self._ID < 1 or self._ID > 5:
                        self._status = self.idErr
                    else:
                        if self._ID == 1:
                            self.getDataSub1()
                        elif self._ID == 2:
                            self.getDataSub2()
                        elif self._ID == 3:
                            self.getDataSub3()
    #                    elif self._ID == 4:
    #                        self.getDataSub4()
    #                    elif self._ID == 5:
    #                        self.getDataSub5()
                        self._status = self.noErr
        return self._status

    def getDataSub1(self):
        self._weekNum = self.BinToInt(self._words[2,:10])
        self._satAcc = self.BinToInt(self._words[2,12:16])
        self._satHealth = self.BinToInt(self._words[2,16:22])
        self._IODC = self.BinToInt(np.append(self._words[2,22:24],
                                   self._words[7,:8]))
        self._Tgd = self.BinToInt(self._words[6,16:24],
                                  signed=True)*2**(-31)                # in s
        self._Toc = self.BinToInt(self._words[7,8:24])*16              # in s
        self._af2 = self.BinToInt(self._words[8,0:8],
                                  signed=True)*2.0**(-55)              # s/s^2
        self._af1 = self.BinToInt(self._words[8,8:24],
                                  signed=True)*2.0**(-43)              # in s/s
        self._af0 = self.BinToInt(self._words[9,0:22],
                                  signed=True)*2.0**(-31)              # in s

    def getDataSub2(self):
        self._IODE2 = self.BinToInt(self._words[2,0:8])
        self._Crs = self.BinToInt(self._words[2,8:24],
                                  signed=True)*2.0**(-5)               # in m
        self._deltaN = self.BinToInt(self._words[3,0:16],
                                     signed=True)*2.0**(-43)*GPS_PI    # rad/s
        self._M0 = self.BinToInt(np.append(self._words[3,16:24],
                                 self._words[4,0:24]),
                                 signed=True)*2.0**(-31)*GPS_PI        # in rad
        self._Cuc = self.BinToInt(self._words[5,0:16],
                                  signed=True)*2.0**(-29)              # in rad
        self._e = self.BinToInt(np.append(self._words[5,16:24],
                                self._words[6,0:24]))*2**(-33)         # no unit
        self._Cus = self.BinToInt(self._words[7,0:16],
                                  signed=True)*2.0**(-29)              # in rad
        self._sqrtA = self.BinToInt(np.append(self._words[7,16:24],
                                    self._words[8,0:24]))*2.0**(-19)   # m^1/2
        self._Toe = self.BinToInt(self._words[9,0:16])*16              # in s

    def getDataSub3(self):
        self._Cic = self.BinToInt(self._words[2,0:16],
                                  signed=True)*2.0**(-29)              # in rad
        self._omegaBig = self.BinToInt(np.append(self._words[2,16:24],
                                       self._words[3,0:24]),
                                       signed=True)*2.0**(-31)*GPS_PI  # in rad
        self._Cis = self.BinToInt(self._words[4,0:16],
                                  signed=True)*2.0**(-29)              # in rad
        self._i0 = self.BinToInt(np.append(self._words[4,16:24],
                                 self._words[5,0:24]),
                                 signed=True)*2.0**(-31)*GPS_PI        # in rad
        self._Crc = self.BinToInt(self._words[6,0:16],
                                  signed=True)*2.0**(-5)               # in m
        self._omegaSmall = self.BinToInt(np.append(self._words[6,16:24],
                                         self._words[7,0:24]),
                                         signed=True)*2.0**(-31)*GPS_PI # rad
        self._omegaDot = self.BinToInt(self._words[8,0:24],
                                       signed=True)*2.0**(-43)*GPS_PI  # rad/s
        self._IDOT = self.BinToInt(self._words[9,8:22],
                                   signed=True)*2.0**(-43)*GPS_PI      # rad/s
        self._IODE3 = self.BinToInt(self._words[9,0:8])

#    def getDataSub4(self):
#        return 0

#    def getDataSub5(self):
#        return 0

    def CheckParity(self):
        res = 0
        i = 1
        while res == 0 and i < 10:
            DS29 = self._words[i-1,28]
            DS30 = self._words[i-1,29]
            d = self._words[i,:24]
            if DS30 == 1:
                d = 1 - d
                self._words[i,:24] = d
            D25 = DS29^d[ 0]^d[ 1]^d[ 2]^d[ 4]^d[ 5]^d[ 9]^d[10]\
                      ^d[11]^d[12]^d[13]^d[16]^d[17]^d[19]^d[22]
            D26 = DS30^d[ 1]^d[ 2]^d[ 3]^d[ 5]^d[ 6]^d[10]^d[11]\
                      ^d[12]^d[13]^d[14]^d[17]^d[18]^d[20]^d[23]
            D27 = DS29^d[ 0]^d[ 2]^d[ 3]^d[ 4]^d[ 6]^d[ 7]^d[11]\
                      ^d[12]^d[13]^d[14]^d[15]^d[18]^d[19]^d[21]
            D28 = DS30^d[ 1]^d[ 3]^d[ 4]^d[ 5]^d[ 7]^d[ 8]^d[12]\
                      ^d[13]^d[14]^d[15]^d[16]^d[19]^d[20]^d[22]
            D29 = DS30^d[ 0]^d[ 2]^d[ 4]^d[ 5]^d[ 6]^d[ 8]^d[ 9]\
                      ^d[13]^d[14]^d[15]^d[16]^d[17]^d[20]^d[21]^d[23]
            D30 = DS29^d[ 2]^d[ 4]^d[ 5]^d[ 7]^d[ 8]^d[ 9]^d[10]\
                      ^d[12]^d[14]^d[18]^d[21]^d[22]^d[23]
            if (np.array([D25,D26,D27,D28,D29,D30],dtype=np.int8) \
            != self._words[i,24:]).any():
                res = i
            i += 1
        return res

    # signed as two's complement
    def BinToInt(self,bits,signed=False):
        z = 0
        f = 1
        neg = signed and bits[0]==1
        if neg:
            bits = 1-bits
        for b in reversed(bits):
            z += int(b)*f
            f *= 2
        if neg:
            z = -(z+1)
        return z


# ---------------------------- Class SatPos() ----------------------------
# calculates orbit parameter and position of a satellite for a given time
# Result: (X,Y,Z), orbit parameters and corrections of satellite time.
# Based on GPS Standard Positioning Signal Specification, 2nd Edition 1995

class SatPos():
    muE = 3.986005E+14        # m^3/s^2; Earth's universal gravitational
                              # parameter, WGS84
    OmE = OMEGA_EARTH         # rad/sec; Earth's rotation rate
                              # (angular velocity), WGS84
    F   = -4.44280763310E-10  # s/sqrt(m);  = -2 sqrt(mue)/c^2

    def __init__(self):
        self.dctOrb = {}
        self.dctTime = {}

    # Following quantities are calculated here:
    #
    # A     :   Semi-major axis
    # tk    :   Time from ephemeris reference epoch,
    #           t is GPS system time at time of transmission,
    #           gps_toe is the epoc time
    # Mk    :   Corrected mean anomaly
    # Ek    :   Eccentric anomaly
    # nuk   :   True anomaly
    # Phik  :   Argument of latitude
    # duk   :   Argument of latitude correction
    # rk    :   Corrected radius
    # drk   :   Further radius correction
    # ik    :   Corrected inclination
    # dik   :   Further correction to inclination
    # uk    :   Corrected argument of latitude
    # Omegak:   Corrected longitude of ascending node
    # opxk  :   x position in orbital plane
    # opyk  :   y position in orbital plane
    # xk    :   x position in Earth-Centered, Earth-Fixed coordinates
    # yk    :   x position in Earth-Centered, Earth-Fixed coordinates
    # zk    :   z position in Earth-Centered, Earth-Fixed coordinates
    #
    # Below, variables 'gps_...' are the corresponding quantities given
    # in the subframes

    # accounts for week crossovers
    def CrossTime(self,t):
        halfWeek = WEEK_IN_SEC // 2
        while t > halfWeek:
            t -= 2*halfWeek
        while t < -halfWeek:
            t += 2*halfWeek
        return t

    # nominal time of transmission, "effective SV PRN code phase time";
    # to get the actual gps time, tsv is corrected using dtsv, t=tsv-dtsv
    def tsv(self,gps_tow):
        return (gps_tow-1)*6

    def dtsv(self,t_sv,gps_af0,gps_af1,gps_af2,gps_toc,gps_tgd,dtr=0):
        res = gps_af0+gps_af1*self.CrossTime(t_sv-gps_toc)\
              +gps_af2*self.CrossTime(t_sv-gps_toc)**2+dtr-gps_tgd
        return res

    # exact GPS system time at time of transmission; here only from beginning
    # of week epoch (Sunday, 0:00)
    def gpst(self,t_sv,dt_sv):
        return t_sv - dt_sv

    # time from ephemeris reference epoch; difference between gps time and
    # the epoch time toe must account for  beginning or end of week crossovers
    def tk(self,gps_t,gps_toe):
        t_k = self.CrossTime(gps_t - gps_toe)
        return t_k

    # Semi-major axis
    def A(self,gps_sqrtA):
        return gps_sqrtA**2

    # Corrected mean anomaly
    def Mk(self,t_k,gps_sqrtA,gps_deltaN,gps_M0):
        return gps_M0 + (np.sqrt(self.muE)/gps_sqrtA**3+gps_deltaN)*t_k

    # eccentric anomaly E in rad;
    # Kepler equation: M=E-e*sin(E) with mean anomaly M and eccentricity e
    def Ek(self,M_k,gps_e,itMax=10,eps=1.0E-12):
        q0 = 0
        E_k = M_k
        it = 0
        while abs(E_k-q0) > eps and it < itMax:
            q0 = E_k
            E_k = q0-(q0-gps_e*np.sin(q0)-M_k)/(1-gps_e*np.cos(q0))
            it += 1
        return E_k

    # True anomaly
    def nuk(self,E_k,gps_e):
        return np.arctan2(np.sqrt(1-gps_e**2)*np.sin(E_k),(np.cos(E_k)-gps_e))

    # Argument of latitude
    def Phik(self,nu_k,gps_omegaSmall):
        return nu_k + gps_omegaSmall

    # Argument of latitude correction
    def duk(self,Phi_k,gps_cus,gps_cuc):
        return gps_cus*np.sin(2*Phi_k)+gps_cuc*np.cos(2*Phi_k)

    # Corrected radius
    def drk(self,Phi_k,gps_crc,gps_crs):
        return gps_crc*np.cos(2*Phi_k)+gps_crs*np.sin(2*Phi_k)

    # Correction to inclination
    def dik(self,Phi_k,gps_cic,gps_cis):
        return gps_cic*np.cos(2*Phi_k)+gps_cis*np.sin(2*Phi_k)

    # Corrected argument of latitude
    def uk(self,Phi_k,d_uk):
        return Phi_k + d_uk

    # Corrected radius
    def rk(self,E_k,d_rk,gps_sqrtA,gps_e):
        return gps_sqrtA**2 * (1-gps_e*np.cos(E_k))+d_rk

    # Corrected inclination
    def ik(self,t_k,d_ik,gps_i0,gps_idot):
        return gps_i0 + d_ik + gps_idot*t_k

    # x position in orbital plane
    def opxk(self,r_k,u_k):
        return r_k*np.cos(u_k)

    # y position in orbital plane
    def opyk(self,r_k,u_k):
        return r_k*np.sin(u_k)

    # Corrected longitude of ascending node
    def Omegak(self,t_k,gps_omegaBig,gps_omegaDot,gps_toe):
        return gps_omegaBig + (gps_omegaDot-self.OmE)*t_k - self.OmE*gps_toe

    # x positiion in Earth-Centered, Earth-Fixed coordinates
    def xk(self,op_xk,op_yk,i_k,Omega_k):
        return op_xk*np.cos(Omega_k) - op_yk*np.cos(i_k)*np.sin(Omega_k)

    # x positiion in Earth-Centered, Earth-Fixed coordinates
    def yk(self,op_xk,op_yk,i_k,Omega_k):
        return op_xk*np.sin(Omega_k) + op_yk*np.cos(i_k)*np.cos(Omega_k)

    # z position in Earth-Centered, Earth-Fixed coordinates
    def zk(self,op_yk,i_k):
        return op_yk*np.sin(i_k)

    # Calculation of relevant coordinates for parameters given in subframes;
    # eph is a dictionary for the orbital parameters from the subframes 1,2,3

    def gpsUTCStr(self,gps_tow,eph):
        gps_t = self.gpsTime(gps_tow,eph)
        d = datetime.datetime(1980,1,6) \
            +datetime.timedelta(seconds=round(gps_t-LEAPSEC))

        return d.strftime('%a, %d.%m.%Y %H:%M:%S UTC')

    def gpsTime(self,gps_tow,eph):
        t_sv = self.tsv(gps_tow)
        dt_sv = self.dtsv(t_sv,eph['af0'],eph['af1'],eph['af2'],
                               eph['Toc'],eph['Tgd'])
        gps_t = (eph['weekNum']+ROLLOVER*1024)*WEEK_IN_SEC \
                + self.gpst(t_sv,dt_sv)
        return gps_t


    def ecefCoord(self,gps_tow,eph,DT=0,relCorr=True):

        itRelCorr = 2 if relCorr else 1
        t_sv = self.tsv(gps_tow) + DT
        t_gd = eph['Tgd']
        t_oc = eph['Toc']
        t_oe = eph['Toe']
        dtr = 0
        for it in range(itRelCorr):
            dt_sv = self.dtsv(t_sv,eph['af0'],eph['af1'],eph['af2'],
                              t_oc,t_gd,dtr=dtr)
            gps_t = self.gpst(t_sv,dt_sv)
            t_k = self.tk(gps_t,t_oe)
            M_k = self.Mk(t_k,eph['sqrtA'],eph['deltaN'],eph['M0'])
            E_k = self.Ek(M_k,eph['e'])
            if itRelCorr and it==0:
                dtr = self.F*eph['e']*eph['sqrtA']*np.sin(E_k)

        nu_k  = self.nuk(E_k,eph['e'])
        Phi_k = self.Phik(nu_k,eph['omegaSmall'])
        d_ik  = self.dik(Phi_k,eph['Cic'],eph['Cis'])
        i_k   = self.ik(t_k,d_ik,eph['i0'],eph['IDOT'])
        d_uk = self.duk(Phi_k,eph['Cus'],eph['Cuc'])
        u_k = self.uk(Phi_k,d_uk)
        d_rk = self.drk(Phi_k,eph['Crc'],eph['Crs'])
        r_k = self.rk(E_k,d_rk,eph['sqrtA'],eph['e'])
        op_xk = self.opxk(r_k,u_k)
        op_yk = self.opyk(r_k,u_k)

        Omega_k = self.Omegak(t_k,eph['omegaBig'],eph['omegaDot'],eph['Toe'])
        X = self.xk(op_xk,op_yk,i_k,Omega_k)
        Y = self.yk(op_xk,op_yk,i_k,Omega_k)
        Z = self.zk(op_yk,i_k)

        self.dctOrb = {'Omega_k': Omega_k,'M_k': M_k,'E_k': E_k,'nu_k': nu_k,
                       'Phi_k': Phi_k,'d_ik': d_ik,'i_k': i_k,'d_uk': d_uk,
                       'u_k': u_k,'d_rk': d_rk,'r_k': r_k,'op_xk': op_xk,
                       'op_yk': op_yk}

        self.dctTime = {'tsv': t_sv,   # code phase time (ideally: GPS time)
                        'dtsv': dt_sv, # correction to tsv; clock inaccuracies
                                       # and propagation delay to antenna
                        'dtr': dtr,    # relativistic correction
                        'gpst': gps_t, # GPS time; only since last week epoch
                        'tk': t_k}     # for orbit calculation; time from
                                       # ephemeris reference epoch
                        #'Tgd': t_gd,  # group time delay, parameter for dtsv
                        #'Toe': t_oe,  # epoch time; reference time ephemerides
                                       # ("offset eph")
                        #'Toc': t_oc,  # parameter for "offset code phase" dtsv
                                       # time from ephemeris reference epoch
                                       # for calculating satellite orbit

        return X,Y,Z,dt_sv

# -------------------------- Class SatData() --------------------------------
# Subframes from a satellite are passed to a SatData() instance to build
# ephemeris and a time table of (tow,ST). A complete ephemeris table from
# subframes 1 to 3 is built by reading many subframes. The validity of the
# subframes are checked and a possible change of ephemeris data in the stream
# of subframes is monitored. The time data tow and ST from all subframes 1-5
# are saved in a table.

class SatData():
    errlst = ['no error',
              'not yet ready',
              'new ephemerides',
              'flawed frame',
              'not healthy']
    noErr = 0
    notReady = 1
    newEphem = 2
    flawedFrame = 3
    healthErr = 4

    def __init__(self,satNo,ephemeris=None):
        self._satNo = satNo
        self._status = 0
        self._ephemData = {}
        self._timeData = []
        self._ephemOk = False
        self._SF1 = False
        self._SF2 = False
        self._SF3 = False
        self._SFLst = []
        self._IODC  = -1
        self._IODE2 = -1
        self._IODE3 = -1
        self._Health = -1
        self._Accuracy = -1            # currently not used
        self._lastIODC = -1
        self._lastTow = -1
        self._lastST = -1
        self.EPHEM_LOADED = (ephemeris is not None)
        if self.EPHEM_LOADED:
            self.loadEphem(ephemeris)

    @property
    def errmsg(self):
        return self.errlst[self._status]

    @property
    def status(self):
        return self._status

    @property
    def ephemOk(self):
        return self._ephemOk

    @property
    def ephemData(self):
        return self._ephemData

    @property
    def timeData(self):
        return self._timeData

    @property
    def subFrameLst(self):
        return self._SFLst

    @property
    def IODC(self):
        return self._IODC

    def loadEphem(self,eph):
        self._ephemData = eph.copy()
        self._ephemData['SAT'] = self._satNo
        self._ephemOk = True
        self._SF1 = True
        self._SF2 = True
        self._SF3 = True
        self._IODC  = eph['IODC']
        self._IODE2 = eph['IODE2']
        self._IODE3 = eph['IODE3']
        self._Accuracy = eph['satAcc']
        self._Health = 0

        self._lastIODC = self._IODC & 255


    def framesValid(self,subframe):
        status = self.noErr
        iodc = -1
        if subframe['ID'] == 1:
            iodc = subframe['IODC'] & 255
            self._Health = subframe['satHealth']
            if self._Health != 0:
                status = self.healthErr
        elif subframe['ID'] == 2:
            iodc = subframe['IODE2']
        elif subframe['ID'] == 3:
            iodc = subframe['IODE3']

        if status == self.noErr and iodc > -1:
            if self._lastIODC > -1 and iodc != self._lastIODC:
                status = self.newEphem
            self._lastIODC = iodc

        self._lastTow = subframe['tow']
        self._lastST = subframe['ST']

        return status


    def readSubframe(self,subframe,saveSubframe = False):
        if saveSubframe:
            self._SFLst.append(subframe)
        self._status = self.framesValid(subframe)   # checks also iodc

        if self._status == self.noErr:              # no update of ephem
            if not self._ephemOk:
                if subframe['ID'] == 1 and not self._SF1:
                    for key in ephemSF1:
                        self._ephemData[key]=subframe[key]
                    self._IODC = subframe['IODC']
                    self._Accuracy = subframe['satAcc']
                    self._SF1=True
                elif subframe['ID'] == 2 and not self._SF2:
                    for key in ephemSF2:
                        self._ephemData[key]=subframe[key]
                    self._IODE2 = subframe['IODE2']
                    self._SF2=True
                elif subframe['ID'] == 3 and not self._SF3:
                    for key in ephemSF3:
                        self._ephemData[key]=subframe[key]
                    self._IODE3 = subframe['IODE3']
                    self._SF3=True
                self._ephemOk = self._SF1 and self._SF2 and self._SF3
                self.EPHEM_LOADED = False

            # Reading an out-dated ephemeris from file can cause an error if
            # subframe 4 or 5 set here a time reference since neither IODC nor
            # weekNum can be checked to ensure a valid ephemeris.
            # EPHEM_LOADED=true involves that ephemOk=True.
            if (self._ephemOk and not self.EPHEM_LOADED) or \
               (self.EPHEM_LOADED and subframe['ID'] < 4):
                self._timeData.append((subframe['tow'],subframe['ST']))

        return self._status


# ----------- Class SatOrbit() ------------------------------------------------
# Input for the function readFrame() are dictionairies each containing a single
# subframe. Many succeeding subframes are passed to an instance of SatData() to
# build an ephemeris table and a reference time table. A SatPos() instance is
# used to evaluate this data and to calculate the positions of the satellite for
# corrected GPS times using the time table. Feeding the function evalCodePhase()
# with a list of code phases for given stream numbers outputs a list of tuples
# with satellite positions (x,y,z) at corrected satellite time (tow,n*N_CYC*1ms)
# and corrected receiving time ST.

class SatOrbit():
    errlst = ['no error',
              'not ready',
              'new ephem',
              'flawed',
              'unhealthy']
    noErr = 0
    notReady = 1
    newEphem = 2
    flawedFrame = 3
    healthErr = 4
    maxSlope = 6.55E-3                # in units of sample time, 3.2ns/ms;
                                      # max change of adjacent code phases

    def __init__(self,num,eph=None):
        self._satNo = num
        self._status = 0
        self.data = SatData(num,ephemeris=eph)
        self.position = SatPos()
        self._datLst = []             # backup of previous data (not used)
        self.cpLst = []               # corrected code phase list
        self.lastSNO = 0              # last stream number
        self.lastCP = 0               # last code phase
        self.REF_TIME = None          # active time reference
        self.REF_EPHEM = None         # active ephemeris
        self.PHASE_ERR = []           # stream numbers with phase error
        self.SCPLst = []              # list of cp curve slopes per N_CYC*1ms
        self.maxSCPLst = 1024//N_CYC  # max length of SCPLst for averaging
        self.minSCPLst = 4            # min length for averaging

    @property
    def errmsg(self):
        return self.errlst[self._status]

    @property
    def status(self):
        return self._status

    @property
    def datLst(self):
        return self._datLst

    @property
    def satNo(self):
        return self._satNo


    def readFrame(self,subframe):
        streamNo = subframe['ST'] // NGPS
        if len(self.PHASE_ERR) > 0 and streamNo < self.PHASE_ERR[-1]:
            self._status = self.flawedFrame
        else:
            self._status = self.data.readSubframe(subframe,saveSubframe=False)
            if self._status == self.newEphem:
                if self.data.ephemOk:
                    self._datLst.append(self.data) # backup previous data
                self.data = SatData(self._satNo)   # new SatData instance
                self.data.readSubframe(subframe,saveSubframe=False)
                                                   # read the subframe again

        return self._status


    def getStdDev(self,tcpLst,cophLst):
        if len(cophLst) > 3:
            p = np.polyfit(tcpLst,cophLst,1)       # calc stdDev by subtracting
                                                   # a linear fit from the data
            fit = np.poly1d(p)
            cophStd = np.std(cophLst-fit(tcpLst))
            self.SCPLst.append(p[0]/N_CYC)         # scale slope to change/ms
            if len(self.SCPLst) > self.maxSCPLst:
                del self.SCPLst[0]
        else:
            cophStd = 0.5
            fit = None
        cophStd *= GPS_C/SAMPLE_RATE               # standard deviation in m
        meanSlope = 0
        if len(self.SCPLst) > self.minSCPLst:
            meanSlope = np.mean(self.SCPLst)       # average over ~1000 slopes
        if abs(meanSlope) > self.maxSlope:
            meanSlope = np.sign(meanSlope)*self.maxSlope  # limit meanSlope

        return cophStd,meanSlope


    def clearCodePhaseRef(self):
        self.lastSNO = 0
        self.cpLst = []
        self.SCPLst = []
        self.REF_TIME = None
        self.REF_EPHEM = None


    # code phase list was already checked regarding phase errors and overflow;
    # see cpOflCorrection in gpseval.py
    # cpl is list of tuples (streamNo, codePhase)
    def evalCodePhase(self,cpl,relCorr=True):
        minGap = 1000                            # in units of N_CYC*1ms
        maxGap = 10000                           # arbitrary; might be extended
                                                 # max cp drift is 6.6/s
        minFitNo = N_CYC // 2                    # min no of entries for fit
        maxFitNo = 100
        diffTol = 200                            # tolerance; depends on maxGap

        result = []
        if len(cpl) > 0:
            if cpl[0][1] is None:                # phase error occured
                self.PHASE_ERR.append(cpl[0][0]) # save error to exclude frames
                self.data._timeData = []         # time ref invalid;
                self.clearCodePhaseRef()         # reset (TOW,ST)
                return result
            cpl = list(filter(lambda item: item[0] > self.lastSNO,cpl))
                                                 # exclude doubles
        # replace REF_TIME by a new one
        if self.REF_TIME is not None and self.data.ephemOk \
        and self.data.ephemData['IODC'] != self.REF_EPHEM['IODC']:
            self.clearCodePhaseRef()

        # timeData != [] involves that data.ephemOk=True and
        # that iodc has been checked for valid ephemeris
        if self.REF_TIME is None and len(self.data._timeData) > 0:
            self.REF_TIME = self.data._timeData[-1]  # (TOW,ST)
            self.REF_EPHEM = self.data.ephemData.copy()

        if len(cpl) == 0 or self.REF_TIME is None:
            return result

        weekNum = self.REF_EPHEM['weekNum']
        TOW,ST = self.REF_TIME                   # from last valid subframe
        ST_DEL = ST % CODE_SAMPLES               # integer next to code phase
        ST = (ST // CODE_SAMPLES) * CODE_SAMPLES # get rid of delay to add later
                                                 # interpolated code phase
        ST_SNO = ST // NGPS                      # stream number

        # ST contains three pieces of information: stream number sno=ST//NGPS,
        # position within stream p=ST//CODE_SAMPLES-sno*N_CYC, and code phase
        # (as integer) with delay = ST % CODE_SAMPLES. Currently code phases
        # typically arrive < 6s before the respective ST, since the validity of
        # a subframe can be checked only 6s after the preamble arrived.
        # If the code phase steps over 2047 and thus jumps to 0 (or reverse
        # from 0 to 2047) WITHIN a stream, this was corrected already before.
        # If the reference ST is not changed, the subsequent lists of code
        # phases must also be corrected regarding this overflow. This is done
        # by comparison with the lastly gathered code phase lastCP. It is
        # expected that differences of neighbored code phases (even with
        # missing streams) are below diffTol.

        if ST_SNO > self.lastSNO:
            self.lastSNO = ST_SNO
            self.lastCP = ST_DEL

        cpl_SNO,cpl_CP = zip(*cpl)
        cpl_CP = np.asarray(cpl_CP)

        # check overflow by using a linear fit to previous cplst in case of a
        # large gap of stream numbers
        if  cpl_SNO[0]-self.lastSNO > maxGap:      # if gap too large for fit
            self.clearCodePhaseRef()               # reset ref time
            return result
        elif cpl_SNO[0]-self.lastSNO > minGap:
            if len(self.cpLst) >= minFitNo:
                x,y = zip(*self.cpLst[-maxFitNo:]) # linear fit; max. curvature
                p = np.polyfit(x,y,1)              # of 0.7 Hz/s leads to
                fit = np.poly1d(p)                 # mismatch of < diffTol
                self.lastCP = fit(cpl_SNO[0])
            else:                                  # reset ref time if not
                self.clearCodePhaseRef()           # enough points for fit
                return result

        lastOfl = self.lastCP // CODE_SAMPLES
        if lastOfl != 0:
            cpl_CP += lastOfl*CODE_SAMPLES

        # an overflow occured if diff > diffTol, so correct it
        diff = self.lastCP - cpl_CP[0]
        if np.isclose(abs(diff),CODE_SAMPLES,rtol=1E-5,atol=diffTol):
            cpl_CP += np.sign(diff)*CODE_SAMPLES

        # cophStd ist used as weight in least-squares fit
        # slopeCP is used to correct CP regarding the position in stream
        cophStd,slopeCP = self.getStdDev(cpl_SNO,cpl_CP)

        cpl = list(zip(cpl_SNO,cpl_CP))
        self.cpLst += cpl
        self.lastSNO,self.lastCP = cpl[-1]
        # start offset for given TOW in ms
        offms = (TOW % 2**(N_CYC // 32)) * 16 if N_CYC > 16 else 0

        # increase TOW (time) up to first stream given in list
        while (ST + 6*SAMPLE_RATE)//NGPS < cpl_SNO[0]:
            ST += 6*SAMPLE_RATE
            TOW += 1
            offms = (offms + 16) % N_CYC

        CP = cpl_CP[0]
        cycNo = 0
        # general: deltaST = (offms + cycNo*N_CYC)*CODE_SAMPLES
        deltaST = offms*CODE_SAMPLES
        streamNo = (ST+deltaST)//NGPS
        # position of code within stream 0 .. N_CYC-1
        codeNo = (ST+deltaST) // CODE_SAMPLES - streamNo*N_CYC
        idx = 0
        while idx < len(cpl_SNO):
            if cpl_SNO[idx] < streamNo:       # should not happen
                idx += 1
            elif cpl_SNO[idx] > streamNo:
                streamNo += 1
                cycNo += 1
                deltaST += NGPS
            else:
                x,y,z,d_st = self.position.ecefCoord(
                             TOW,self.REF_EPHEM,
                             DT=deltaST/SAMPLE_RATE,
                             relCorr=relCorr)
                CP = cpl_CP[idx]

                # correction due to position shift regarding
                # CP measurement in center of stream
                corrCP = (codeNo+CP//CODE_SAMPLES-N_CYC//2)*slopeCP
                smpTime = (ST+deltaST+CP+corrCP)/SAMPLE_RATE + d_st  # in s
                result.append((self._satNo,TOW,x,y,z,
                               smpTime,weekNum,cycNo,cophStd))
                streamNo += 1
                cycNo += 1
                deltaST += NGPS
                idx += 1

            if deltaST >= 6*SAMPLE_RATE:
                TOW += 1
                cycNo = 0
                ST += 6*SAMPLE_RATE
                offms = (offms + 16) % N_CYC
                deltaST = offms*CODE_SAMPLES
                # REF_TIME is changed to avoid initial increment of ST;
                # CP ist needed in case of errors
                if streamNo < cpl_SNO[-1]:
                    self.REF_TIME = (TOW,ST + CP % CODE_SAMPLES)

        return result


# -------- Class SatStream() -------------

class SatStream():
    PREAMBLE = np.array([1,-1,-1,-1,1,-1,1,1],dtype=np.int8)
    DF_GAIN1 = 10                                # PLL gain before phaseLock
    DF_GAIN2 = 1                                 # PLL gain after phaseLock
    MIN_CORR_Q = -0.9                            # threshold for sweep trigger

    def __init__(self,satNo,freq,itSweep=10,corrMin=8,\
                 corrAvg=8,sweepCorrAvg=4,delay=0):
        self.SAT_NO = satNo
        self.SEC_TIME = np.linspace(1,NGPS,NGPS,endpoint=True,\
                                    dtype=MY_FLOAT)/SAMPLE_RATE
        # first entry of EDGES is sign of edge, then (MS_TIME,SMP_TIME)
        self.EDGES = [0]
        self.PHASE_LOCKED = False
        self.PHASE = 0.0
        self.FREQ = freq
        self.GPSBITS = np.array([],dtype=np.int8)     # bit array of +1 and -1
        self.GPSBITS_ST = np.array([],dtype=np.int64) # SMP_TIME of edges
        self.PREV_SAMPLES = []
        self.MS_TIME = 0
        self.SMP_TIME = 0
        self.FFT_CACODE = fft(GPSCacode(satNo))
        self.DELAY = delay
        self.NO_SEC = 1024 // N_CYC              # no of streams in 1 sec
        self.NO_BEFORE_CALC = self.NO_SEC        # eval & send subframes
                                                 # once in a second
        self.CORR_MIN = corrMin
        self.CORR_AVG = min(corrAvg,N_CYC)
        self.SWEEP_CORR_AVG = sweepCorrAvg
        self.CACODE_REP = GPSCacodeRep(satNo,N_CYC,0)
        self.STD_DEV = 0.005                     # overwritten by 1st stream
        self.AMPLITUDE = 0.0
        self.MAX_CORR = 0.0
        self.SWEEP = False
        self.IT_SWEEP = itSweep
        self.PREV_STREAM_NO = 0
        self.PREV_SIGNAL = 0                     # used in decodeData
        self.DF_NO = self.NO_SEC                 # no of entries in DF
        self.DF = [0]                            # drift of doppler frequeny
        self.CORR_Q = 0                          # -1..+1; mean of corr results
                                                 # of last 60 sec (CORRLST_NO)
        self.CORR_L = 0                          # ditto, but only last second
        self.CORRLST_NO = 60*self.NO_SEC         # no of entries in CORRLST
        self.CORRLST = [0]                       # codePhase correlations
        self.REP_SWEEP = False                   # report sweep in sent data
        self.CALC_PLOT = False                   # calc plot data of amplitude
        self.GPSDATA = []                        # measured amplitude
        self.BITDATA = []                        # amplitude from evaluation

    # ---------------

    def erasePrevData(self):
        self.EDGES = [0]
        self.GPSBITS = np.array([],dtype=np.int8)
        self.GPSBITS_ST = np.array([],dtype=np.int64)
        self.PREV_SAMPLES = []


    def setPhaseUnlocked(self):
        self.PHASE_LOCKED = False
        self.CORRLST = [0]
        self.MS_TIME = 0
        self.PHASE = 0.0
        self.erasePrevData()


    def initSweep(self):
        self.setPhaseUnlocked()
        self.FREQ_SAVE = self.FREQ               # backup if sweep fails
        self.DF_SAVE = self.DF.copy()
        self.FREQ = MIN_FREQ
        self.DF = [0]
        self.SWEEP = True

    def restoreFreq(self):
        self.FREQ = self.FREQ_SAVE
        self.DF = self.DF_SAVE.copy()

    # ---------------

    def reportValues(self,frameLst):
        for dct in frameLst:
            dct['SAT'] = self.SAT_NO
            dct['AMP'] = self.AMPLITUDE
            dct['CRM'] = self.MAX_CORR
            dct['FRQ'] = self.FREQ
            dct['SWP'] = self.REP_SWEEP
        self.REP_SWEEP = False


    def checkCorrQuality(self):
        sweep = False
        if len(self.CORRLST) >= self.CORRLST_NO: # min of 60 s between sweeps
            sweep = (self.CORR_Q < self.MIN_CORR_Q)
        return sweep


    def process(self,data,smpTime,sweep=False):
        self.SMP_TIME = smpTime
        streamNo = smpTime // NGPS
        if streamNo -1 != self.PREV_STREAM_NO:
            self.erasePrevData()                 # a data stream was skipped
        self.PREV_STREAM_NO = streamNo
        sweep = sweep and not self.SWEEP         # ignore trigger if sweep
                                                 # is already running
        if sweep:
            self.initSweep()                     # initialize sweep and
                                                 # set SWEEP=True

        if self.SWEEP:                           # start or continue sweep
            self.REP_SWEEP = self.SWEEP          # for report in frameLst

            # A call of sweepFrequencies() tries IT_SWEEP different
            # frequencies in steps of STEP_FREQ. SWEEP is set False if a
            # correlation was found (delay > 0) or entire range FREQ_MIN
            # to FREQ_MAX was tested in several calls without success.
            self.SWEEP,self.FREQ,self.MAX_CORR,\
            delay,codePhase = self.sweepFrequency(data,self.FREQ)
            self.CORR_Q,self.CORR_L = self.corrQuality(codePhase)

            if delay >= 0:                      # sweep was successful
                self.DELAY = delay
            elif not self.SWEEP:                # all frequencies failed
                 self.restoreFreq()             # restore initial values

            # send report with frameLst once in a second
            frameLst = []
            if streamNo % self.NO_BEFORE_CALC == 0:
                frameLst = [{}]
                self.reportValues(frameLst)

        else:
            # demodulate signal using current doppler freq and phase,
            # then calculate delay and interpolated code phase by correlation
            data,self.PHASE = self.demodDoppler(data,self.FREQ,self.PHASE,NGPS)
            _,delay,codePhase,normMaxCorr = self.cacodeCorr(data,self.CORR_AVG)
            self.CORR_Q,self.CORR_L = self.corrQuality(codePhase)
            if delay >= 0:                       # True if correlation > min
                self.DELAY = delay

            # gpsData (amplitude) has N_CYC complex data points (1 per ms)
            gpsData = self.decodeData(data,self.DELAY)
            self.STD_DEV = np.std(np.abs(gpsData))
            self.AMPLITUDE = np.mean(np.abs(gpsData))/self.STD_DEV
            self.MAX_CORR = normMaxCorr

            # Process and send data once in a second
            frameLst = []
            if streamNo % self.NO_BEFORE_CALC == 0:
                if self.PHASE_LOCKED:
                    frameLst = self.evalEdges()  # list of subframe data
                if len(frameLst) == 0:
                    frameLst = [{}]
                self.reportValues(frameLst)      # add further information
                sweep = self.checkCorrQuality()

            # A bad signal triggers sweep for next call of process(),
            # otherwise phase and frequency are adjusted in PLL
            if sweep:
                self.initSweep()
            else:
                dfreq,phaseshift,\
                self.PHASE_LOCKED,phase = self.phaseLockedLoop(gpsData)
                self.PHASE += phaseshift
                self.FREQ  = self.confineRange(self.FREQ + dfreq)

        return self.SWEEP,frameLst,codePhase,(self.CORR_Q,self.CORR_L)


    # ---------------

    def phaseLockedLoop(self,gpsData):
        # Feedback control keeps the phase close to 0, so that the imaginary
        # part of the amplitude is small. This is achieved by calculating
        # the absolute phase change as a function of time from the complex
        # amplitude. The phase is then used to correct the doppler frequency
        # and to subtract the total phaseOffset. The tracking of frequency is
        # facilitated by subtracting the almost constant frequency drift using
        # an averaged value (meanDF). The max. drift is ~2.6Hz/s. An upper
        # limit for the frequency correction df is set to maxDF=20Hz/s. This
        # prevents unlocking in case of bad signals. Smaller values slow down
        # the initial phase locking.
        phOffAvg = 4                    # average no for phase offset
        minDiff = 2.0                   # if phase jump > minDiff, add pi
        lockedThres = 0.1               # threshold for locked state
        maxDF = 20/self.NO_SEC          # =20Hz/s; limit for frequency change

        phaseIsLocked = self.PHASE_LOCKED
        n = len(gpsData)
        phase = np.arctan(gpsData.imag/gpsData.real)

        dp = 0
        realPhase = np.copy(phase)
        for i in range(1,n):            # rebuild absolute phase change
            delta = phase[i]-phase[i-1]
            if abs(delta) > minDiff:
                dp -= np.sign(delta)
            realPhase[i] += dp*np.pi

        phaseOffset = np.mean(realPhase[-phOffAvg:])
        phaseDev = np.mean(realPhase)

        if phaseIsLocked:
            meanDF = np.mean(self.DF)
            fbDF = self.DF_GAIN2*phaseDev
            df = fbDF + meanDF
            if abs(df) > maxDF:
                df = np.sign(df)*maxDF
            if len(self.DF) >= self.DF_NO:
                del self.DF[0]
            self.DF.append(df)
        else:
            df = self.DF_GAIN1*phaseDev
            self.DF = [df]

        if abs(phaseDev) < lockedThres:
            phaseIsLocked = True

        return df, phaseOffset, phaseIsLocked, realPhase


    # ---------------


    def fitCodePhase(self,gpsCorr,mx):
        # gpsCorr[mx] is correlation maximum
        lgc = len(gpsCorr)
        ma = mx - 1 if mx > 0 else lgc-1
        mb = mx + 1 if mx < lgc-1 else 0

        # Fit to symmetric triangle; tip (or maximum) is at tmx
        # range from -0.5 to +0.5
        if gpsCorr[ma] > gpsCorr[mb]:
            tmx = 0.5*(gpsCorr[mb]-gpsCorr[ma])/(gpsCorr[mx]-gpsCorr[mb])
        else:
            tmx = 0.5*(gpsCorr[mb]-gpsCorr[ma])/(gpsCorr[mx]-gpsCorr[ma])

        # Fit to parabola; maximum of parabola is at pmx
        # range from -0.5 to +0.5
        pmx = 0.5 * (gpsCorr[mb]-gpsCorr[ma]) \
                  / (2*gpsCorr[mx]-gpsCorr[mb]-gpsCorr[ma])

        # mean of both fits gives best value (from tests)
        # possible range is from -0.5 to 2047.5
        fitMax = mx + 0.5*(tmx+pmx)

        return fitMax


    def findCodePhase(self,gpsCorr):
        mean = np.mean(gpsCorr)
        std = np.std(gpsCorr)
        delay = -1
        codePhase = -1.0
        mx = np.argmax(gpsCorr)
        normMaxCorr = (gpsCorr[mx]-mean)/std
        if normMaxCorr > self.CORR_MIN:
            delay = mx
            codePhase = self.fitCodePhase(gpsCorr,mx)

        return delay,codePhase,normMaxCorr

    # ---------------

    # The position of the correlaton peak determined here depends on which of
    # the FFT-transformed arrays (fftData or FFT_CACODE) is conjugated for the
    # back transformation. Here, if the position of the first bit of the cacode
    # sequence in the data is shifted to the right by DS, the correlation peak
    # is at DELAY = DS, which is the returned value in correlation. Perfect
    # alignment is achieved if the C/A code is rolled to the right by DELAY=DS,
    # np.roll(cacode,+DELAY). This is done in decodeData() for decoding.
    def cacodeCorr(self,data,corrAvg):
        df = 0
        nc = len(data)//CODE_SAMPLES
        p = (nc-corrAvg) // 2
        # correlation is done with data in center of stream
        for i in range(p,p+corrAvg):
            dfm = fft(data[i*CODE_SAMPLES:(i+1)*CODE_SAMPLES])
            df += dfm
        fftData = df/corrAvg
        fftCorr = fftData*np.conjugate(self.FFT_CACODE)
        corr = np.abs(ifft(fftCorr))
        delay,codePhase,normMaxCorr = self.findCodePhase(corr)
        return corr,delay,codePhase,normMaxCorr

    # ---------------

    def corrQuality(self,codePhase):
        cpq = -1 if codePhase < 0 else 1
        self.CORRLST.append(cpq)
        if len(self.CORRLST) > self.CORRLST_NO:
            del self.CORRLST[0]
        corrQ = np.mean(self.CORRLST)
        corrLast = np.mean(self.CORRLST[-self.NO_SEC:])

        return corrQ,corrLast

    # ---------------

    def demodDoppler(self,data,freq,phase,N):
        factor = np.exp(-1.j*(phase+2*np.pi*freq*self.SEC_TIME[:N]))
        phase += 2*np.pi*freq*self.SEC_TIME[N-1]
        return factor*data[:N], np.remainder(phase,2*np.pi)

    #------------------

    def getCorrMax(self,data,corrAvg,freq):
        phase = 0
        N = corrAvg*CODE_SAMPLES
        new1,_ = self.demodDoppler(data,freq,phase,N)
        corr1,delay,coPh,normMaxCorr = self.cacodeCorr(new1,corrAvg)
        if delay > -1:
            max1 = corr1[delay]
        else:
            max1 = 0
        return delay,max1,coPh,normMaxCorr


    def sweepFrequency(self,data,freq):
        sweepFreq = True
        j = 0
        delay = -1
        coPh = -1
        while delay < 0 and j < self.IT_SWEEP:
            delay,maxCorr,coPh,normMaxCorr = self.getCorrMax(\
                                             data,self.SWEEP_CORR_AVG,freq)
            if delay < 0:
                freq = freq+STEP_FREQ
            j += 1

        if delay >= 0:
            sweepFreq = False
        elif freq > MAX_FREQ:
            freq = MIN_FREQ
            sweepFreq = False

        return sweepFreq,freq,normMaxCorr,delay,coPh


    # ---------------

    # In the correlation a delay (integer code phase) was determined ranging
    # from 0 to 2047. The C/A code repeat, which has the same length as the
    # data stream, is rolled to the right by delay and then multiplied with
    # the stream to decode the data. The start bit of the C/A code in the data
    # array is at SMP_TIME+delay. In PREV_SAMPLES, it is at position SMP_TIME+
    # delay-2048, if delay has not changed. To reduce the noise, the 2048
    # values in the range of each C/A code are averaged and the mean values are
    # saved in gpsData (array of length N_CYC). Incomplete data at the end
    # having a length < len(C/A code) are saved in PREV_SAMPLE for next call.
    def decodeData(self,data,delay):
        MIN_EDGE_AMP = 3*self.STD_DEV            # minimum amplitude change
                                                 # to pass as edge event
        # determine last sign in EDGES
        prevSign = (2*(len(self.EDGES) % 2) - 1) * self.EDGES[0]

        cacode = np.roll(self.CACODE_REP,delay)
        y = cacode*data

        NPS = len(self.PREV_SAMPLES)
        if NPS > 0:
            y = np.append(self.PREV_SAMPLES,y)
        NS = NGPS + NPS

        n0 = 0
        n1 = NPS + delay
        # for first avg sometimes n1-n0 != 2048; n=NPS is at data[0]
        if n1 == 0:
            n1 = CODE_SAMPLES
            ST = self.SMP_TIME
        else:
            ST = self.SMP_TIME + delay - CODE_SAMPLES  # sample time (local)

        gpsData = []
        while n1 <= NS:
            m = np.mean(y[n0:n1])
            gpsData.append(m)
            if self.PHASE_LOCKED:
                mSign = np.sign(m.real)
                if self.EDGES[0] == 0:           # initially no entries
                    self.EDGES[0] = mSign        # first entry is sign
                    prevSign = mSign             # of first signal
                else:
                    # EDGES saves timestamps of signal changes (sign) as tuple
                    # (MS_TIME,SAMPLE_TIME), where MS_TIME is satellite time
                    # and SAMPLE_TIME is local time (SDR clock)
                    if mSign != prevSign \
                    and prevSign*self.PREV_SIGNAL > 0 \
                    and abs(m.real-self.PREV_SIGNAL) > MIN_EDGE_AMP:
                        self.EDGES.append((self.MS_TIME,ST+n0))
                        prevSign = mSign
                self.PREV_SIGNAL = m.real
                self.MS_TIME += 1                # len(C/A code) equates to 1ms
            n0 = n1
            n1 += CODE_SAMPLES
        gpsData = np.asarray(gpsData,dtype=MY_COMPLEX)
        self.PREV_SAMPLES = y[n0:NS]             # save partial C/A codes for
                                                 # use in next call
        if self.CALC_PLOT:
            self.GPSDATA = gpsData
            self.BITDATA = self.bitPlotData(len(gpsData))

        return gpsData

    # ---------------


    def evalEdges(self):
        frameData = []

        if len(self.EDGES) > 2:
            bits,bitsSmpTime = self.logicalBits()
            self.GPSBITS = np.append(self.GPSBITS,bits)
            self.GPSBITS_ST = np.append(self.GPSBITS_ST,bitsSmpTime)
            frameData,self.GPSBITS,\
            self.GPSBITS_ST = self.evalGpsBits(self.GPSBITS,self.GPSBITS_ST)

        return frameData

    # ---------------

    def logicalBits(self):
        bits = []
        bitsSmpTime = []
        lastSign = self.EDGES[0]
        n = len(self.EDGES)

        if n > 2:
            t1,st1 = self.EDGES[1]
            for i in range(2,n):
                t2,st2 = self.EDGES[i]
                m,r = np.divmod(t2-t1,20)        # bit has length of 20ms
                if r > 17:
                    m += 1
                if m > 0:
                    bits += [lastSign]*m
                    bitsSmpTime += [st1]         # save sample time of edge
                                                 # for use with preamble
                    bitsSmpTime += [0]*(m-1)     # fill array so that it
                                                 # has same length as bits
                t1 = t2
                st1 = st2
                lastSign = -lastSign
            self.EDGES = [lastSign,self.EDGES[-1]] # delete evaluated data

        bits = np.asarray(bits,dtype=np.int8)    # array of +1 and -1 values
        bitsSmpTime = np.asarray(bitsSmpTime,dtype=np.int64)

        return bits, bitsSmpTime


    #------------- Function evalGpsBits (uses Subframe) -----------------------
    #
    # Read a stream of bits (gpsBits with +1 and-1), find the locations of
    # the preamble, and extract the data using an instance of Subframe().
    # The data of the input arrays not decoded in this process are returned as
    # result together with a dictionairy containing the subframe parameter. The
    # sample times ST of the preambles are entered in the subframe dictionairy
    # as additional entry.

    def evalGpsBits(self,gpsBits,gpsBitsSmpTime):
        Result = []
        if len(gpsBits) < 300:
            return Result, gpsBits, gpsBitsSmpTime

        gb = np.copy(gpsBits)
        bitsCorr = np.correlate(gb,self.PREAMBLE,mode='same')
        locPreamble =[]
        for i in range(len(bitsCorr)):
            if abs(bitsCorr[i])==8:
                locPreamble.append(i-4)         # begin of preamble is 4 bits
                                                # before maximum of correlation
        start = 0
        if len(locPreamble) > 0:
            gb[gb==-1] = 0                      # convert -1 to logical 0
            lpIndex = 0
            start = locPreamble[lpIndex]
            ok = True
            while ok and start+300 < len(gb):   # ok=True if start was changed
                sf = Subframe()                 # at the end of the loop
                if sf.Extract(gb[start:start+300]) == 0:
                    ST = gpsBitsSmpTime[start]
                    if sf.ID == 1:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'weekNum': sf.weekNum,
                               'satAcc': sf.satAcc,
                               'satHealth': sf.satHealth,
                               'Tgd': sf.Tgd,
                               'IODC': sf.IODC,
                               'Toc': sf.Toc,
                               'af2': sf.af2,
                               'af1': sf.af1,
                               'af0': sf.af0,
                               'ST': ST}
                    elif sf.ID == 2:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'Crs': sf.Crs,
                               'deltaN': sf.deltaN,
                               'M0': sf.M0,
                               'Cuc': sf.Cuc,
                               'IODE2': sf.IODE2,
                               'e': sf.e,
                               'Cus': sf.Cus,
                               'sqrtA': sf.sqrtA,
                               'Toe': sf.Toe,
                               'ST': ST}
                    elif sf.ID == 3:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'Cic': sf.Cic,
                               'omegaBig': sf.omegaBig,
                               'Cis': sf.Cis,
                               'i0': sf.i0,
                               'IODE3': sf.IODE3,
                               'Crc': sf.Crc,
                               'omegaSmall': sf.omegaSmall,
                               'omegaDot': sf.omegaDot,
                               'IDOT': sf.IDOT,
                               'ST': ST}
                    elif sf.ID == 4 or sf.ID == 5:
                        res = {'ID': sf.ID,
                               'tow': sf.tow,
                               'ST': ST}
                    Result.append(res)
                    start += 300
                else:
                    ok = False
                    while not ok and lpIndex<len(locPreamble)-1:
                        lpIndex += 1
                        s = locPreamble[lpIndex]
                        ok = s > start
                    if ok:
                        start = s

        return Result, gpsBits[start:], gpsBitsSmpTime[start:]


    # Produce a non-periodic pulse train in steps of 1ms from the data
    # in EDGES. if no error occured it follows the course of amplitude
    # in gpsData, however, without noise. n is the requested length of data
    # (typically n = N_CYC) and it is assumed that the current MS_TIME is
    # the last point in the amplitude array. The data is used in a plot for
    # comparision with the measured data.
    def bitPlotData(self,n):
        bd = np.zeros(n,dtype=np.int8)
        t1 = self.MS_TIME
        t0 = t1 - n + 1
        firstSign = self.EDGES[0]

        if len(self.EDGES) == 1:
            bd[:] = firstSign
        elif len(self.EDGES) == 2:
            t,st = self.EDGES[1]
            if t >= t0:
                bd[:t-t0] = firstSign
                bd[t-t0:n] = -firstSign
            else:
                bd[:] = -firstSign
        else:
            lastSign = (2*(len(self.EDGES) % 2) - 1) * self.EDGES[0]
            bsc = [tms for tms,st in self.EDGES[1:]]
            k1 = len(bsc)
            k0 = k1-1
            while k0>0 and bsc[k0]>t0:
                k0 -= 1
            ts = 0
            for t in reversed(bsc[k0:k1]):
                te = min(ts+t1-t, n)
                bd[ts:te] = lastSign
                ts = te
                t1 = t
                lastSign = -lastSign
            if te < n:
                bd[te:n] = lastSign

        bd = np.flip(bd)

        return bd


    def confineRange(self,freq):
        if freq > MAX_FREQ:
            freq = MAX_FREQ
        elif freq < MIN_FREQ:
            freq = MIN_FREQ
        return freq


# -------- end class SatStream() ----------


# ------- Positioning by means of the Newton-Gauss algorithm ----------------


# for 4 or more satellites
def JacobianCalc(pos,satPos,rangeEst):
    jacob = np.zeros((np.size(satPos,1),4))
    jacob[:,0] = -np.ones(np.size(satPos,1))       # part. deriv. reg. c*t0
    jacob[:,1] = (pos[1] - satPos[0,:])/rangeEst   # part. deriv. reg (x,y,z)
    jacob[:,2] = (pos[2] - satPos[1,:])/rangeEst
    jacob[:,3] = (pos[3] - satPos[2,:])/rangeEst

    return jacob

# for 3 or more satellites
def JacobianCalc3(pos,satPos,rangeEst):
    a = 6378137                                     # in m; WGS84 major axis
    f = 1/298.257223563                             # WGS84 flattening:
                                                    # f = (a-b)/a
    ab2 = 1/(1-f)**2                                # (a/b)^2
    absPos = np.sqrt(pos[1]**2+pos[2]**2+ab2*pos[3]**2)
    jacob = np.zeros((np.size(satPos,1)+1,4))
    jacob[:-1,0] = -np.ones(np.size(satPos,1))      # part. deriv. reg. c*t0
    jacob[:-1,1] = (pos[1] - satPos[0,:])/rangeEst  # part. deriv. reg (x,y,z)
    jacob[:-1,2] = (pos[2] - satPos[1,:])/rangeEst
    jacob[:-1,3] = (pos[3] - satPos[2,:])/rangeEst
    jacob[-1,0] = 0
    jacob[-1,1] = pos[1]/absPos
    jacob[-1,2] = pos[2]/absPos
    jacob[-1,3] = ab2*pos[3]/absPos

    return jacob


def rotEarth(recPos,rangeEst):
    dt = rangeEst/GPS_C
    v = [-recPos[2]*OMEGA_EARTH,recPos[1]*OMEGA_EARTH,0]
    dp = np.tensordot(v,dt,0)

    return dp

# see https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
# and https://en.wikipedia.org/wiki/Weighted_least_squares
#
# Position of receiver and propagation time t0 of Satellite 0 is calculated
# from known satellite positions and measured time delays between subframes
# number of satellites, i.e. dimension of satPos, must be n >= 4.
#
# satPos is np.array([x,y,z],:noSats) in ECEF coordinates
# recPos is receiver position [t0,x,y,z] to be determined

def leastSquaresPos(minSat,satPos,timeDelay,**kwargs):
    if minSat == 4:
        recPos,residLst,rangeEst,measDelay = leastSquaresPos4(\
                                                satPos,timeDelay,**kwargs)
    else:
        recPos,residLst,rangeEst,measDelay = leastSquaresPos3(\
                                                satPos,timeDelay,**kwargs)

    return recPos,residLst,rangeEst,measDelay


def leastSquaresPos4(satPos,timeDelay,recPos=[0,0,0,0],maxResidual=1.0E-8,\
                     maxIt=10,t0Guess=0.07,height=150,hDev=1,stdDev=None):

    dt = timeDelay - timeDelay[0]
    cdt = GPS_C*dt
    recPos[0] = GPS_C*t0Guess
    residLst = []
    dp = np.zeros((3,len(dt)))

    if stdDev is None:
        W = np.eye(len(dt))
    else:
        W = np.linalg.inv(np.diag(stdDev)**2)    # weight matrix: 1/sigma^2 in
                                                 # diagonal; sigma in m
    it = 0
    residual = 1
    while it < maxIt and residual > maxResidual:
        rangeEst = np.sqrt( (satPos[0,:]-recPos[1]-dp[0,:])**2 \
                          + (satPos[1,:]-recPos[2]-dp[1,:])**2 \
                          + (satPos[2,:]-recPos[3]-dp[2,:])**2)

        # displacement of receiver position due to rotation of earth
        dp = rotEarth(recPos,rangeEst)

        # function to minimize; free variables are in recPos
        fgn = rangeEst - recPos[0] - cdt

        Jac = JacobianCalc(recPos,satPos,rangeEst)
        JacT = Jac.T

        deltaPos = -np.linalg.pinv(JacT.dot(W).dot(Jac)).dot(JacT).dot(W) @ fgn
        recPos += deltaPos

        residual = np.linalg.norm(deltaPos)
        residLst.append(residual)
        it += 1

    measDelay = cdt + recPos[0]                  # propagation times (in m)

    return recPos,residLst,rangeEst,measDelay


# for 3 or more Satellites; z-position pos[3] is fixed by x^2+y^2+z^2 = R^2
# with R given by earth radius; earth ellipsoid is (x^2+y^2)/a^2 + z^2/b^2 = 1
def leastSquaresPos3(satPos,timeDelay,recPos=[0,3687000,3687000,0],\
                     maxResidual=1.0E-8,maxIt=10,t0Guess=0.07,height=150,\
                     hDev=1,stdDev=None):
    a = 6378137
    f = 1/298.257223563
    ab2 = 1/(1-f)**2

    dt = timeDelay - timeDelay[0]
    cdt = GPS_C*dt
    recPos[0] = GPS_C*t0Guess
    recPos[3] = (1-f)*np.sqrt((a+height)**2-recPos[1]**2-recPos[2]**2)
    recPos = np.asarray(recPos)
    fgn = np.zeros(len(dt)+1)
    residLst = []
    dp = np.zeros((3,len(dt)))

    if stdDev is None:
        W = np.eye(len(dt)+1)
    else:
        stdDev = np.append(stdDev,[hDev])
        W = np.linalg.inv(np.diag(stdDev)**2)

    it = 0
    residual = 1
    while it < maxIt and residual > maxResidual:
        rangeEst = np.sqrt( (satPos[0,:]-recPos[1]-dp[0,:])**2 \
                          + (satPos[1,:]-recPos[2]-dp[1,:])**2 \
                          + (satPos[2,:]-recPos[3]-dp[2,:])**2)

        # displacement of recorder position due to rotation of earth
        dp = rotEarth(recPos,rangeEst)

        # functions to minimize; free variables are in pos
        fgn[:-1] = rangeEst - recPos[0] - cdt
        fgn[-1] = np.sqrt(recPos[1]**2+recPos[2]**2+ab2*recPos[3]**2)\
                  -(a+height)

        Jac = JacobianCalc3(recPos,satPos,rangeEst)
        JacT = Jac.T

        deltaPos = -np.linalg.pinv(JacT.dot(W).dot(Jac)).dot(JacT).dot(W) @ fgn
        recPos += deltaPos

        residual = np.linalg.norm(deltaPos)
        residLst.append(residual)
        it += 1

    measDelay = cdt + recPos[0]                  # propagation times (in m)

    return recPos,residLst,rangeEst,measDelay



# ------ Transformation ECEF coordinates to geodetic coordinates -------------
# (x,y,z) --> (latitude,longitude, altitude)
# Karl Osen. Accurate Conversion of Earth-Fixed Earth-Centered Coordinates to
# Geodetic Coordinates. [Research Report] Norwegian University of Science and
# Technology. 2017. hal-01704943v2
# half as fast as pyproj.transformer.transform(), but more accurate
#
# short names
invaa    = +2.45817225764733181057e-0014   # 1/(a^2)
aadc     = +7.79540464078689228919e+0007   # (a^2)/c
bbdcc    = +1.48379031586596594555e+0002   # (b^2)/(c^2)
l        = +3.34718999507065852867e-0003   # (e^2)/2
p1mee    = +9.93305620009858682943e-0001   # 1-(e^2)
p1meedaa = +2.44171631847341700642e-0014   # (1-(e^2))/(a^2)
Hmin     = +2.25010182030430273673e-0014   # (e^12)/4
ll4      = +4.48147234524044602618e-0005   # e^4
ll       = +1.12036808631011150655e-0005   # (e^4)/4
invcbrt2 = +7.93700525984099737380e-0001   # 1/(2^(1/3))
inv3     = +3.33333333333333333333e-0001   # 1/3
inv6     = +1.66666666666666666667e-0001   # 1/6
d2r      = +1.74532925199432957691e-0002   # pi/180
r2d      = +5.72957795130823208766e+0001   # 180/pi

def geoToEcef(lat,lon,alt):
    lat = d2r * lat
    lon = d2r * lon
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)
    N = aadc/np.sqrt(coslat * coslat + bbdcc)
    d = (N + alt) * coslat
    x = d * coslon
    y = d * sinlon
    z = (p1mee * N + alt) * sinlat
    return x,y,z


def ecefToGeo(coordinates):
    x,y,z = coordinates
    ww = x * x + y * y
    m = ww * invaa
    n = z * z * p1meedaa
    mpn = m + n
    p = inv6 * (mpn - ll4)
    G = m * n * ll
    H = 2 * p * p * p + G
    if (H < Hmin):
        return None
    C = math.pow(H + G + 2 * np.sqrt(H * G), inv3) * invcbrt2
    i = -ll - 0.5 * mpn
    P = p * p
    beta = inv3 * i - C - P / C
    k = ll * (ll - mpn)

    t1 = beta * beta - k
    t2 = np.sqrt(t1)
    t3 = t2 - 0.5 * (beta + i)
    t4 = np.sqrt(t3)

    t5 = np.abs(0.5 * (beta - i))
    # t5 may accidentally drop just below zero due to numeric turbulence
    # This only occurs at latitudes close to +- 45.3 degrees
    t6 = np.sqrt(t5)
    t7 = t6 if (m < n) else -t6

    t = t4 + t7
    # Use Newton-Raphson's method to compute t correction
    j = l * (m - n)
    g = 2 * j
    tt = t * t
    ttt = tt * t
    tttt = tt * tt
    F = tttt + 2 * i * tt + g * t + k
    dFdt = 4 * ttt + 4 * i * t + g
    dt = -F / dFdt
    # compute latitude (range -PI/2..PI/2)
    u = t + dt + l
    v = t + dt - l
    w = np.sqrt(ww)
    zu = z * u
    wv = w * v
    lat = np.arctan2(zu, wv)
    # compute altitude
    invuv = 1 / (u * v)
    dw = w - wv * invuv
    dz = z - zu * p1mee * invuv
    da = np.sqrt(dw * dw + dz * dz)
    alt = -da if (u < 1) else da
    # compute longitude (range -PI..PI)
    lon = np.arctan2(y, x)

    lat = r2d * lat
    lon = r2d * lon

    return lat,lon,alt


# ----- Transformation ECEF coordinates to local azimuth & elevation ----------
#
# input: observer position [x,y,z] (ECEF), satellite position [x,y,z]
# output: elevation & azimuth (theta,phi) in degree (north = 0, east = 90° etc)
def ecefToAzimElev(obsPos,satPos):
    r1    = np.asarray(obsPos)           # vector to observer
    mr1   = np.sqrt(np.dot(r1,r1))       # magnitude of r1
    r2    = np.asarray(satPos)           # vector to satellite
    r21   = r2 - r1                      # difference vector
    mr21  = np.sqrt(np.dot(r2-r1,r2-r1)) # magnitude of r2-r1
    n1    = r1 / mr1                     # norm. vector of local area at obsPos
    r21p  = np.dot(n1,r2-r1)*n1          # projection of r2-r1 to n1
    r21e  = r2 - r1 - r21p               # projection of r2-r1 to area
    mr21e = np.sqrt(np.dot(r21e,r21e))   # magn. of projection r2-r1 to area
    z1    = np.asarray([0,0,1])          # direction 'north'
    z1p   = np.dot(z1,n1)*n1             # projection of z1 to n1
    z1e   = z1 - z1p                     # projection of z1 to area
    mz1e  = np.sqrt(np.dot(z1e,z1e))     # magn. of projection of z1 to area

    # scalar product of n1 and r21/mr21 is sine of elevation angle theta, thus
    theta = np.arcsin(np.dot(n1,r21)/mr21)/np.pi*180

    # scalar product of z1e and r21e gives azimuth angle phi, thus
    phi = np.arccos(np.dot(z1e,r21e)/(mz1e*mr21e))/np.pi*180

    # triple product (Spatprodukt) determines sign of angle
    if np.dot(n1,np.cross(r21e,z1e)) < 0:
        phi = -phi

    return theta,phi


# ------ calc distances from local position in geodetic coordinates -----------
# based on ellipsoidal model of earth (WGS84)
# input:  lonHome, latHome is local position (in degree)
#         lon,lat is in a (small) distance to "Home" (in degree)
# output: delta x in m (along latitude to east),
#         delty y in m (along longitude to north)
# geodetic coordinates are (lat,lon,height)
def locDistFromLatLon(geoHome,geoPos):
    eqAxis =  6378137.0
    flat =  0.003352810
    latHome,lonHome,_ = geoHome
    lat,lon,_ = geoPos
    lonDistPerDeg = eqAxis*(np.sin(latHome/180*np.pi)**2\
                    +((1-flat)*np.cos(latHome/180*np.pi))**2)**(3/2)/(1-flat)\
                    *np.pi/180
    latDistPerDeg = eqAxis*np.cos(latHome/180*np.pi)*np.pi/180

    return (lon-lonHome)*latDistPerDeg, (lat-latHome)*lonDistPerDeg

# ------ time string from GPS week number and time-of-week --------------------

def gpsTime(tow,weekNum):
    # numpy types are converted to native python
    tow = getattr(tow, "tolist", lambda: tow)()
    weekNum = getattr(weekNum, "tolist", lambda: weekNum)()
    # tow is valid for start of next subframe; tow-1 for current subframe
    d = datetime.datetime(1980,1,6) \
        + datetime.timedelta(days=(weekNum+ROLLOVER*1024)*7) \
        + datetime.timedelta(seconds=(tow-1)*6-LEAPSEC)

    return d


def gpsTimeStr(tow,weekNum,timeOnly = False):
    d = gpsTime(tow,weekNum)
    if timeOnly:
        return d.strftime('%H:%M:%S UTC')
    else:
        return d.strftime('%a, %d.%m.%Y %H:%M:%S UTC')



