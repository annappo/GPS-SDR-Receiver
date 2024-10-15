# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

from gpsglob import \
    SHOW_MAP_HTML,MARKER_SIZE,MARKER_COL,TRACK_COL,\
    TRACK_WEIGHT,TRACK2_COL,TRACK2_WEIGHT,SCALE_THRES
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,CheckButtons,TextBox
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import gpslib
import numpy as np
import datetime
import webbrowser
from  pathlib import Path
import sys
import folium

# a fixed color for each PRN number from 1 to 32 taken from CSS4_COLORS
SAT_COL_LST = [None,'lightgray','blue','brown','cyan','darkblue','crimson',
              'darkred','black','orange','magenta','red','yellow','darkgreen',
              'gray','violet','darkgrey','olive','lightgreen','blueviolet',
              'chocolate','gold','darkorange','greenyellow','darkgoldenrod',
              'sienna','teal','wheat','pink','aquamarine','orangered','salmon',
              'tan']
MIN_SAT_NO = 1
MAX_SAT_NO = len(SAT_COL_LST) - 1

# ----- class GpsUI for user interface ----------------

class GpsUI():
    winTitle = 'GPS Tracking using RTL-SDR & Python'
    satCol = SAT_COL_LST
    minSatNo = MIN_SAT_NO
    maxSatNo = MAX_SAT_NO
    
    def __init__(self,figSize,outDist,plotScale=200,autoScale=True,
                 posSize=3,height=0,confHeight=False):
        # variables for callback functions
        self.geoMeanPosLat = 0
        self.geoMeanPosLon = 0
        self.plotCenter = None
        self.trackEvent = None
        self.trigStop = False
        self.trigClose = False
        self.trigSweep = False
        self.trigShowMap = False
        self.resetStats = False
        self.zoom = 1                           # 1, 0.5, or 2
        self.txtHeight = str(height)
        self.confHeight = confHeight

        self.satList = []                       # satellite list in polar plot
        self.legHnd = []                        # legend handle for polar plot
        self.satTracks = {}                     # tracks in local coordinates
        self.startTime = 0                      # time of first position fix
        self.mask = {}                          # saves refs of plot and events
        self.vdW,self.vdH = 750,150             # virtual text display W x H
        self.vdL = (self.vdH-np.asarray(
                    range(0,170,10)))/self.vdH  # general line numbers
        self.vdMaxSat = 12                      # max satellite no for display
        self.secTrack = None                    # for track in browser

        # figure and axes for plots
        self.fig = plt.figure(figsize=figSize)
        spec = self.fig.add_gridspec(2,2)
        self.axTxt = self.fig.add_subplot(spec[0, :])
        self.axTxt.set_axis_off()
        self.axPos = self.fig.add_subplot(spec[1, 0])
        self.axSat = self.fig.add_subplot(spec[1, 1],projection='polar')
        self.fig.canvas.header_visible = False
        self.fig.canvas.manager.set_window_title(self.winTitle)
        # initialize plots & Events
        self.initInfoText()
        self.initStatsText(outDist)
        self.axPos.callbacks.connect('xlim_changed',self.on_xlim_change)
        self.axPos.callbacks.connect('ylim_changed',self.on_ylim_change)
        self.initTrackPlot(plotScale,posSize)        # range is 2*plotScale
        self.mapZoom = self.calcMapZoom(2*plotScale) # for track on map
        self.initSatPlot()
        self.initButtons(autoScale)
        mpl.rcParams['keymap.save'] = ['ctrl+s']
        mpl.rcParams['keymap.quit'] = []
        self.fig.canvas.mpl_connect('key_press_event', self.onKeypress)
        # bring window to foregrund; block is False for interactive mode on
        plt.ion()
        plt.show()

    def initTrackPlot(self,plotScale,posSize):
        # plot for tracking positions
        self.axPos.axis('equal')
        trackX = [0]
        trackY = [0]
        posL, = self.axPos.plot(trackX,trackY,'ob',ms = 0.1)
        self.axPos.set_xlim(-plotScale,+plotScale)  # triggers on_xlim_change
        self.axPos.set_ylim(-plotScale,+plotScale)  # triggers on_ylim_change
        self.axPos.grid(ls = '--')
        self.mask['trk'] = posL
        posL, = self.axPos.plot(trackX,trackY,'-or',ms=posSize,lw=0.1)
        self.mask['sectrk'] = posL


    def initInfoText(self):
        # text output - satellite information
        xs = np.asarray([0,50,100,150,220,280,310])/self.vdW   # tabs for data

        startL = self.axTxt.text(xs[0],self.vdL[0],'')
        self.mask['start'] = startL
        timeL = self.axTxt.text(xs[4]+10/self.vdW,self.vdL[0],'')
        self.mask['time'] = timeL

        keys = ['PRN','AMP','COR','FRQ','TOW','ID','EPH']
        for i,key in enumerate(keys):
            self.axTxt.text(xs[i],self.vdL[2],key)
        dataL = []
        for i in range(self.vdMaxSat):
            dataL.append({})
            for j,key in enumerate(keys):
                dataL[i][key] = self.axTxt.text(xs[j],self.vdL[i+3],'')
        self.mask['data'] = dataL


    def initStatsText(self,outDist):
        # statistics of calculated positions
        xst = np.asarray([400,420,520,570])/self.vdW     # tabs for statistics
        self.axTxt.text(xst[0],self.vdL[0],'ECEF coordinates / m')
        self.axTxt.text(xst[1],self.vdL[2],'MEAN')
        self.axTxt.text(xst[2],self.vdL[2],'SD')
        self.axTxt.text(xst[3],self.vdL[2],'SD of MEAN')
        self.axTxt.text(xst[0],self.vdL[3],'X:')
        self.axTxt.text(xst[0],self.vdL[4],'Y:')
        self.axTxt.text(xst[0],self.vdL[5],'Z:')
        estL = self.axTxt.text(xst[3],self.vdL[0],'')
        meanXL = self.axTxt.text(xst[1],self.vdL[3],'')
        meanYL = self.axTxt.text(xst[1],self.vdL[4],'')
        meanZL = self.axTxt.text(xst[1],self.vdL[5],'')
        sdXL = self.axTxt.text(xst[2],self.vdL[3],'')
        sdYL = self.axTxt.text(xst[2],self.vdL[4],'')
        sdZL = self.axTxt.text(xst[2],self.vdL[5],'')
        sdmXL = self.axTxt.text(xst[3],self.vdL[3],'')
        sdmYL = self.axTxt.text(xst[3],self.vdL[4],'')
        sdmZL = self.axTxt.text(xst[3],self.vdL[5],'')
        self.mask['est'] = estL
        self.mask['mean'] = (meanXL,meanYL,meanZL)
        self.mask['sd'] = (sdXL,sdYL,sdZL)
        self.mask['sdm'] = (sdmXL,sdmYL,sdmZL)

        xmp = np.asarray([400,470,620])/self.vdW     # tabs for mean positions
        self.axTxt.text(xmp[0],self.vdL[7],'MEAN HEIGHT/LAT/LON')
        meanPosHL = self.axTxt.text(xmp[0],self.vdL[8],'')
        meanPosLL = self.axTxt.text(xmp[1],self.vdL[8],'')
        self.mask['meanPos'] = (meanPosHL,meanPosLL)
        
        self.axTxt.text(xmp[0],self.vdL[10],'Skipped data blocks (32ms)')
        self.axTxt.text(xmp[0],self.vdL[11],'Outliers (dr>%1.0f m)'%(outDist))
        self.axTxt.text(xmp[0],self.vdL[12],'LSF failures')
        self.axTxt.text(xmp[0],self.vdL[13],'Phase errors')
        skipL = self.axTxt.text(xmp[2],self.vdL[10],':')
        outlierL = self.axTxt.text(xmp[2],self.vdL[11],':')
        lsfFailL = self.axTxt.text(xmp[2],self.vdL[12],':')
        phaseErrL = self.axTxt.text(xmp[2],self.vdL[13],':')
        self.mask['skip'] = skipL
        self.mask['outlier'] = outlierL
        self.mask['lsfFail'] = lsfFailL
        self.mask['phaseErr'] = phaseErrL
        

    def initSatPlot(self):
        css4 = mcolors.CSS4_COLORS
        # polar plot for satellites
        self.axSat.set_xticks(np.arange(0,2*np.pi,np.pi/4),
                              ['N','NE','E','SE','S','SW','W','NW'])
        self.axSat.set_theta_zero_location("N")
        self.axSat.set_theta_direction('clockwise')
        self.axSat.set_ylim(0,90)
        self.axSat.set_yticks(range(0, 100, 10))
        self.axSat.set_yticklabels(['']+list(map(str, range(80, -10, -10))),
                                             fontsize=6)
        phi = np.asarray([0,.1])
        theta = np.asarray([150,150])  # outside graph
        satL = [None]
        for i in range(self.minSatNo,self.maxSatNo+1):
            col = css4[self.satCol[i]]
            sat1L, = self.axSat.plot(phi,theta,'-',color=col,lw=2,)
            sat2L, = self.axSat.plot(phi[0],theta[0],'o',color=col,ms=4)
            satL.append((sat1L,sat2L))
        self.mask['sat'] = satL
        plt.rcParams["legend.labelspacing"] = 0.2  # default 0.5
        plt.rcParams["legend.handlelength"] = 1.5  # default 2.0


    def initButtons(self,autoScale):
        # define buttons & events
        axSweep = self.fig.add_axes([0.12, 0.95, 0.12, 0.04])
        btSweep = Button(axSweep, 'Sweep')
        btSweep.on_clicked(self.onBtSweep)
        self.mask['sweep'] = (axSweep,btSweep)
        
        axRestart = self.fig.add_axes([0.54, 0.95, 0.16, 0.04])
        btRestart = Button(axRestart, 'Clear Stats & Track')
        btRestart.on_clicked(self.onBtReset)
        self.mask['restart'] = (axRestart,btRestart)
        
        axStop = self.fig.add_axes([0.87, 0.95, 0.12, 0.04])
        btStop = Button(axStop, 'Stop')
        btStop.on_clicked(self.onBtStop)
        self.mask['stop'] = (axStop,btStop)
    
        axMap = self.fig.add_axes([0.87, 0.83, 0.12, 0.04])
        btMap = Button(axMap, 'Show on Map')
        btMap.on_clicked(self.onShowMap)
        self.mask['map'] = (axMap,btMap)

        axConfH = self.fig.add_axes([0.85, 0.71, 0.12, 0.04])
        axConfH.set_frame_on(False)
        btConfH = CheckButtons(axConfH,['Confine Height'],[self.confHeight])
        btConfH.on_clicked(self.onConfHeight)
        self.mask['confH'] = (axConfH,btConfH)
                
        axHeight = self.fig.add_axes([0.93, 0.67, 0.05, 0.04])
        tbHeight = TextBox(axHeight, 'H = ', initial=self.txtHeight,
                           textalignment='left')
        tbHeight.on_submit(self.onTbHeight)
        self.mask['height'] = (axHeight,tbHeight)

        axClose = self.fig.add_axes([0.87, 0.50, 0.12, 0.04])
        btClose = Button(axClose, 'Close')
        btClose.on_clicked(self.onBtClose) 
        self.mask['Close'] = (axClose,btClose)
    
        if autoScale:
            axZoomPlus = self.fig.add_axes([0.49, 0.16, 0.03, 0.04])
            btZoomPlus = Button(axZoomPlus, '+')
            btZoomPlus.on_clicked(self.onZoomPlus)
            self.mask['zoom+'] = (axZoomPlus,btZoomPlus)
    
            axZoomMinus = self.fig.add_axes([0.49, 0.11, 0.03, 0.04])
            btZoomMinus = Button(axZoomMinus, '-')
            btZoomMinus.on_clicked(self.onZoomMinus)    
            self.mask['zoom-'] = (axZoomMinus,btZoomMinus)
            
        # for callback function chooseTrack
        self.trkLines = [self.mask['trk'],self.mask['sectrk']]   
        self.trkLabels = [" all", " 1s"]
        axCheck = self.fig.add_axes([0.49, 0.36, 0.07, 0.1])
        axCheck.set_frame_on(False)
        btCheck = CheckButtons(axCheck, self.trkLabels, [True,True])
        btCheck.on_clicked(self.onCheckTrack)
        self.mask['check'] = (axCheck,btCheck)
        

    def getEvents(self):
        if self.trackEvent is not None:
            index = self.trkLabels.index(self.trackEvent)
            self.trkLines[index].set_visible(
                not self.trkLines[index].get_visible())
            self.trackEvent = None
            
        events = self.trigStop,self.trigClose,self.trigSweep,self.trigShowMap,\
                 self.resetStats,self.zoom        
        self.trigStop,self.trigClose,self.trigSweep = False,False,False
        self.trigShowMap,self.resetStats,self.zoom = False,False,1

        return events,self.confHeight,self.txtHeight


    def updateCanvas(self):
        self.fig.canvas.draw_idle()                      
        self.fig.canvas.flush_events()        
    
                        
    def printTrack(self,locTrack):
        trkX,trkY = zip(*locTrack)
        self.mask['trk'].set_xdata(trkX)
        self.mask['trk'].set_ydata(trkY)
        
            
    def printMeanSecTrack(self,meanSecTrack):
        trkX,trkY = zip(*meanSecTrack)
        self.mask['sectrk'].set_xdata(trkX)
        self.mask['sectrk'].set_ydata(trkY)
        
        
    def setPrintScale(self,loc,zoom):        
        x0,y0 = loc            
        xmin,xmax = self.axPos.get_xlim()
        ymin,ymax = self.axPos.get_ylim()
        dx = (xmax - xmin) / 2
        dy = (ymax - ymin) / 2
        if x0 < xmin or x0 > xmax or y0 < ymin or y0 > ymax or zoom != 1:
            xmin = x0 - zoom*dx
            xmax = x0 + zoom*dx
            ymin = y0 - zoom*dy
            ymax = y0 + zoom*dy
            self.axPos.set_xlim(xmin,xmax)          # triggers on_xlim_change
            self.axPos.set_ylim(ymin,ymax)          # triggers on_ylim_change            
            self.plotCenter = (self.geoMeanPosLat,self.geoMeanPosLon)                        


    # satInfo is list of tuples (satNo,(x,y,z)) for a given tow
    def printSatPos(self,satInfo,recPos):
        css4 = mcolors.CSS4_COLORS
        legChanged = False
        for satNo,satPos in satInfo:
            theta,phi = gpslib.ecefToAzimElev(recPos,satPos)
            if satNo in self.satList:
                prevPhi,prevTheta = self.satTracks[satNo][-1]
                if abs(prevPhi-phi)>0.4 or abs(prevTheta-theta)>0.1:
                    # ~900 pts for full scale 360° or 90°
                    self.satTracks[satNo].append((phi,theta))
                    phiLst,thetaLst = zip(*self.satTracks[satNo])
                    phiLst = np.asarray(phiLst)
                    thetaLst = np.asarray(thetaLst)
                    sat1L,sat2L = self.mask['sat'][satNo]
                    sat1L.set_xdata(phiLst/180*np.pi)
                    sat1L.set_ydata(90-thetaLst)
            elif len(self.satList) < self.vdMaxSat:
                legChanged = True
                self.satList.append(satNo)
                sat1L,sat2L = self.mask['sat'][satNo]
                sat2L.set_xdata([phi/180*np.pi])
                sat2L.set_ydata([90-theta])
                self.satTracks[satNo] = [(phi,theta)]
                col = css4[self.satCol[satNo]]
                line = mlines.Line2D([], [], color=col, lw=3,
                                     label='PRN %02d' % (satNo))
                self.legHnd.append(line)
        
        if legChanged:
            leg = self.axSat.get_legend()
            if leg is not None:
                leg.remove()
            self.axSat.legend(handles=self.legHnd,loc=(1.15,-0.1),
                              fontsize='small')
                    


    def printStat(self,ecefPosStat,geoMeanPos):
        mean,sd,N,out = ecefPosStat
        lat,lon,height = geoMeanPos
    
        self.mask['est'].set_text('(%d estimates)' % (N))
        for i in range(3):
            self.mask['mean'][i].set_text('%9.1f' % (mean[i]))
            self.mask['sd'][i].set_text('%4.1f' % (sd[i]))
            if N > 1:
                self.mask['sdm'][i].set_text('%5.1f' % (sd[i]/np.sqrt(N-1)))
            else:
                self.mask['sdm'][i].set_text('')
            
        self.mask['meanPos'][0].set_text('%1.1f' % (height))
        self.mask['meanPos'][1].set_text('%1.6f°E, %1.6f°N' % (lon,lat))
        
        self.geoMeanPosLat = lat
        self.geoMeanPosLon = lon
        if self.plotCenter is None:
            self.plotCenter=(lat,lon)
            
    
    
    def printSatData(self,frameLst,actSats,errLst,swpLst):        
        frameLst.sort(key = lambda item:item['SAT'])    
        dataL = self.mask['data']
        for i,sf in enumerate(frameLst):
            if i == self.vdMaxSat:
                break
            satNo = sf['SAT']
            dataL[i]['PRN'].set_text('%02d' % (satNo))
            dataL[i]['AMP'].set_text('%4.1f' % (sf['AMP']))
            dataL[i]['COR'].set_text('%4.1f' % (sf['CRM']))
            dataL[i]['FRQ'].set_text('%7.1f' % (sf['FRQ']))
            actStr ='*' if satNo in actSats else ''
            dataL[i]['EPH'].set_text('%s %s' % (sf['EPH'],actStr))
            if 'ID' in sf:
                dataL[i]['ID'].set_text('%d' % (sf['ID']))     
                dataL[i]['TOW'].set_text('%d' % (sf['tow']))
            else:
                dataL[i]['ID'].set_text('')     
                if satNo in errLst:
                    dataL[i]['TOW'].set_text(errLst[satNo])                 
                elif satNo in swpLst:
                    dataL[i]['TOW'].set_text(swpLst[satNo])                 
                else:
                    dataL[i]['TOW'].set_text('')
    
        keys = ['PRN','AMP','COR','FRQ','TOW','ID','EPH']    
        for i in range(len(frameLst),self.vdMaxSat):
            for key in keys:
                dataL[i][key].set_text('')
            
            
    def printTime(self,gpsTime):            
        if gpsTime == 0:
            return
        
        if self.startTime == 0:
            self.startTime = gpsTime
            self.mask['start'].set_text('%s,' % (self.startTime.strftime(
                                        '%a, %d.%m.%y %H:%M:%S UTC')))        
    
        measTime = gpsTime - self.startTime
        totSec = measTime.total_seconds()
        mh, remainder = divmod(totSec, 3600)
        mm, ms = divmod(remainder, 60)    
        self.mask['time'].set_text('DT: %02d:%02d:%02d' % (mh,mm,ms))
        
        
    def printErrors(self,skipData,outlier,lsfFails,noPhaseErr):
        self.mask['skip'].set_text(': %d' % (skipData))
        self.mask['outlier'].set_text(': %d' % (outlier))
        self.mask['lsfFail'].set_text(': %d' % (lsfFails))    
        self.mask['phaseErr'].set_text(': %d' % (noPhaseErr))    
        
        
    def calcMapZoom(self,scale):        
        return int(np.round(26 - np.log2(scale)))
        
    
    def showMap(self,secTrack):
        if self.plotCenter is None or secTrack is None or len(secTrack)==0:
            return
        meanLat,meanLon = self.plotCenter            
        m = folium.Map(location=[meanLat, meanLon],tiles='OpenStreetMap',
                       zoom_start=self.mapZoom,zoom_control=True,
                       control_scale=True)
        coord = [[x,y] for x,y,_ in secTrack]
        if self.trkLines[1].get_visible():
            weight = TRACK_WEIGHT
            color = TRACK_COL
        else:
            weight = TRACK2_WEIGHT
            color = TRACK2_COL
        folium.PolyLine(
            locations=coord,
            smooth_factor=1,
            color=color,
            weight=weight
        ).add_to(m)
        x,y = coord[-1]
        folium.CircleMarker(
            location=[x, y],
            radius=MARKER_SIZE,
            color=MARKER_COL,
            fill_color =MARKER_COL,
            weight=2,
            fill=True,
            fill_opacity=0.2,
            opacity=1,
        ).add_to(m)
        m.save(SHOW_MAP_HTML)
        path = Path.cwd() / SHOW_MAP_HTML
        webbrowser.open('file://%s' % (path.as_posix()))
        
    
    # -------- callback functions for button events ----------------
    
    def onShowMap(self,event):
        self.trigShowMap = True
    
    def onCheckTrack(self,event):
        self.trackEvent = event
    
    def onBtStop(self,event):
        self.trigStop = True
    
    def onBtClose(self,event):
        self.trigClose = True
    
    def onBtSweep(self,event):
        self.trigSweep = True

    def onKeypress(self,event):
        sys.stdout.flush()
        if event.key in ['s','S']:
            self.trigSweep = True 
        elif event.key in ['r','R']:
            self.resetStats = True
        elif event.key in ['q','Q']:
            self.trigStop = True
        elif event.key == '+':
            self.zoom = 0.5
        elif event.key == '-':
            self.zoom = 2
        elif event.key in ['m','M']:
            self.trigShowMap = True
    
    def onBtReset(self,event):
        self.resetStats = True    
        
    def onTbHeight(self,txt):
        try:
            h = int(txt)
        except:
            h = 0
        self.txtHeight = str(h)

    def onConfHeight(self,event):        
        self.confHeight = not self.confHeight
        
    def onZoomPlus(self,event):
        self.zoom = 0.5
        
    def onZoomMinus(self,event):
        self.zoom = 2    
        
        
    #------ scaling of axes -------------
        
    def linScale(self,axmin,axmax):
        diff = axmax-axmin
        z = np.log10(diff)
        ex = int(np.floor(z))
        faktor = 10**ex
        z = diff / faktor + 0.01
        if  z >= 8.0: 
            digit = 0
            scale = 2.0
        elif z >= 4.0:
            digit = 0
            scale = 1.0
        elif z >= 2.4:
            digit = 1
            scale = 0.5
        elif z >= 2.0:
            digit = 1
            scale = 0.4 
        elif z >= 1.5:
            digit = 2
            scale = 0.25
        else: 
            digit = 1
            scale = 0.2            
        scale *= faktor
        digit = digit - ex if digit >= ex else 0
        z = -0.01 if axmin < 0 else 0.99
        tacmin = np.trunc(axmin / scale + z)
        z = -0.99 if axmax < 0 else 0.01
        tacmax = np.trunc(axmax / scale + z)
        ScaleVal = []
        nsv = int(tacmax - tacmin + 1)
        for i in range(nsv):
            sv = (tacmin + i) * scale
            ScaleVal.append(sv)
        
        return np.asarray(ScaleVal),digit        


    def on_xlim_change(self,event_ax):
        xmin,xmax = self.axPos.get_xlim()        
        if xmax-xmin > SCALE_THRES:
            xt,dec = self.linScale(xmin,xmax)
            xtl = 0.001*np.round(xt,dec)
            self.axPos.set_xticks(xt,xtl)
            self.axPos.set_xlabel('Distance (East) / km') 
        else:
            xt,dec = self.linScale(xmin,xmax)
            xtl = np.int32(xt) if dec == 0 else np.round(xt,dec)
            self.axPos.set_xticks(xt,xtl)
            self.axPos.set_xlabel('Distance (East) / m') 
            
        self.mapZoom = self.calcMapZoom(xmax-xmin)
        
        
    def on_ylim_change(self,event_ax):
        ymin,ymax = self.axPos.get_ylim()
        if ymax-ymin > SCALE_THRES:
            yt,dec = self.linScale(ymin,ymax)
            ytl = 0.001*np.round(yt,dec)
            self.axPos.set_yticks(yt,ytl)
            self.axPos.set_ylabel('Distance (North) / km')            
        else:
            yt,dec = self.linScale(ymin,ymax)
            ytl = np.int32(yt) if dec == 0 else np.round(yt,dec)
            self.axPos.set_yticks(yt,ytl)
            self.axPos.set_ylabel('Distance (North) / m')
        
                