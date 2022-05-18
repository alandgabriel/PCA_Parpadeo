# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:46:56 2019

@author: alan_

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import scipy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import scipy.interpolate
import matplotlib 

def aleatorio(semilla):
    a=float(630360016)
    m=float(2147483647)
    return ((a*semilla)%m)/m, (a*semilla)%m

def normal (N,mu,sigma,Z):
    normal=[]
    for i in range (int(N/2)):
        rnd,Z=aleatorio(Z)
        rnd2,Z=aleatorio(Z)
        z=(-2*math.log(rnd))**.5*math.sin(2*math.pi*rnd2)
        z2=(-2*math.log(rnd))**.5*math.cos(2*math.pi*rnd2)      #Box y Muller (1958)
        x1=(z*sigma)+mu
        x2=(z2*sigma)+mu
        normal.append(x1)
        normal.append(x2)
    return normal

def geometric(N,p,Z):
    geometric=[]
    for i in range (N):
        fallidos=0
        rnd=1
        while rnd>p:
            rnd,Z=aleatorio(Z)          #generador de numeros aleatorios con dist geometrica
            fallidos=fallidos+1
        geometric.append(fallidos)
    return geometric

def filtro (sig,wc):
    n   = 4
    fc  = 2*np.array(wc)/fs
    b,a = signal.butter(n,fc,btype = 'bandpass', output='ba')
    sigfilt = signal.filtfilt(b,a,sig)
    return sigfilt

def PCA(A):
    U, Sigma, V = np.linalg.svd(A, full_matrices=False,compute_uv=True)
    X_svd=V
    return X_svd

def nearest_n(x1,x2):
    list2aux = list(x1)
    mylist = []
    for idxlabel in range(0,len(x2)): 
        a = min(enumerate(list2aux), key=lambda x:abs(x[1]-x2[idxlabel]))
        list2aux[a[0]] = 0
        mylist.append(np.copy(a))
    return np.asarray(mylist)


def PSD(x,ts):
    N=len(x)
    rxx=np.convolve(x,np.flip(x))/N
    #rxx=np.correlate(x,x,mode='full')
    sxx=np.fft.fft(rxx)
    mag_sxx=abs(sxx)*ts
    mag_sxx=np.fft.fftshift(mag_sxx)
    return mag_sxx

def barlett_par (v):
    x1=np.arange(0,v/2)
    x2=np.arange(v/2,v)
    triangular=np.append(2*x1/float(v),2-(2*x2/float(v)))
    return triangular
def Periodograma(x,v,labels):
    N=len(x)
    vent=np.floor(N/v)
    f, Pxx_den=scipy.signal.welch(x,fs=fs,window=barlett_par(vent))
    plt.semilogy(f, Pxx_den,label='{0}'.format(labels))
    legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
    legend.get_frame().set_facecolor('pink') 
    plt.xlabel('Hz')
    plt.ylabel('dBs')
    plt.grid()
    
def topomap(signal):
    # some parameters
    N = 300           # number of points for interpolation
    xy_center = [2,2]   # center of the plot
    radius = 2          # radius
    
    # mostly original code
    meanR = signal
    bv=84.98123361344625
    cv=26.1330
    koord=[]
    for i in range(64):
        koord.append([(float(eeg_loc['X'][0,i])+bv)/(bv/2),(float(eeg_loc['Y'][0,i])+bv)/(bv/2)])
        
    x,y = [],[]
    for i in koord:
        x.append(i[0])
        y.append(i[1])
    
    z = meanR
    
    xi = np.linspace(-2, 6, N)
    yi = np.linspace(-2, 6, N)
    zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    
    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"
    
    # make figure
    fig = plt.figure()
    
    # set aspect = 1 to make it a circle
    ax = fig.add_subplot(111, aspect = 1)
    
    # use different number of levels for the fill and the lines
    CS = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)
    ax.contour(xi, yi, zi, 15, colors = "grey", zorder = 2)
    
    # make a color bar
    cbar = fig.colorbar(CS, ax=ax)
    
    # add the data points
    # I guess there are no data points outside the head...
    ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)
    
    # draw a circle
    # change the linewidth to hide the 
    circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none")
    ax.add_patch(circle)
    
    # make the axis invisible 
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
    
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [2,0], width = 0.5, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [2,4], width = 0.5, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    xy = [[3,1.5], [4.5,2],[3,2.5]]
    polygon = matplotlib.patches.Polygon(xy = xy, facecolor = "w", zorder = 0)
    ax.add_patch(polygon) 
        
    # set axes limits
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    
    plt.show() 
    
    
def PCA12(signal,reconstruc):
    eeg_pre=signal[:,idxpre]
    eeg_post=signal[:,idxpost]
    U1, Sigma1, V1 = np.linalg.svd(eeg_pre, full_matrices=False, compute_uv=True)
    principal_components_pre= np.diag(Sigma1[:4]) * np.matrix(V1[:4, :])#V1[:4,:]
    pesos_pre=np.diag(Sigma1[:4]) * np.transpose(np.matrix(U1[:, :4]))
    U2, Sigma2, V2 = np.linalg.svd(eeg_post, full_matrices=False, compute_uv=True)
    principal_components_post=np.diag(Sigma2[:4]) * np.matrix(V2[:4, :])#V2[:4,:]
    pesos_post=np.diag(Sigma2[:4]) * np.transpose(np.matrix(U2[:, :4]))
    plt.figure()
    for i in range (4):
        z=np.zeros(len(eeg_pre[0,:]))
        z2=np.zeros(len(eeg_pre[0,:]))
        for j in range(len(z)):
            z[j]=principal_components_pre[i,j]
            z2[j]=principal_components_post[i,j]
        plt.plot(tiempo[idx],np.append(z,z2),label='PC{0}'.format(i+1))
        legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
        legend.get_frame().set_facecolor('pink') 
        plt.xlabel('tiempo(ms)')
        plt.ylabel('A')
        plt.grid()
    ixc=np.arange(1,5)
    plt.figure()
    for i in range(4):
        z=np.zeros(len(eeg_pre[0,:]))
        z2=np.zeros(len(eeg_pre[0,:]))
        for j in range(len(z)):
            z[j]=principal_components_pre[i,j]
            z2[j]=principal_components_post[i,j]
        plt.subplot(2,2,i+1)
        Periodograma(z,8,'PC{0}_pre'.format(ixc[i]))
        Periodograma(z2,8,'PC{0}_post'.format(ixc[i]))
    for i in range(4):
        z=np.zeros(64)
        z2=np.zeros(64)
        for j in range(len(z)):
            z[j]=pesos_pre[i,j]
            z2[j]=pesos_post[i,j]
        topomap(z)
        topomap(z2)
    if reconstruc:
        Sigma1[0]=0
        Sigma2[0]=0
        x_pre=np.matrix(U1)*np.diag(Sigma1)*np.matrix(V1)
        x_post=np.matrix(U2)*np.diag(Sigma2)*np.matrix(V2)
        x=np.concatenate((x_pre,x_post,signal[:,len(eeg_pre[0,:])*2:640]),axis=1)
        return x
#EJERCICIO 1
registro=scipy.io.loadmat('EEG.mat')
dataeeg=registro['EEG_data']
tiempo=registro['EEG_t'][0,:]
fs=256
ts=float(1)/fs
canal1 ='Oz'
eeg_loc=registro['EEG_chanlocs']
labels_loc=eeg_loc['labels']
ix=(labels_loc==canal1)[0,:]
eeg_ensamblado=np.mean(dataeeg,2)
eeg_ensamblado_canal=eeg_ensamblado[ix,:][0,:]
plt.plot(tiempo,eeg_ensamblado_canal)
plt.ylabel('Amplitud')
plt.xlabel('tiempo (ms)')
plt.title('Canal Oz')
plt.figure()
for i in np.arange(25,36):
    plt.plot(tiempo,eeg_ensamblado[i,:]+(np.mean(eeg_ensamblado[i,:])*20))
    plt.ylabel('Amplitud')
    plt.xlabel('tiempo (ms)')
    plt.title('Todos los canales')
f=plt.figure(figsize=(8,5))
x=eeg_ensamblado_canal
Periodograma(x,10,'Oz')
energ_lat=[]
for i in range(64):
    sig=eeg_ensamblado[i,280:480]
    energ_lat.append(np.dot(sig,sig)/len(sig))
plt.show()
plt.figure()
topomap(energ_lat)


#%% EJERCICIO 2 Y 3

tpreon=np.arange(-1000,0,4)
tposton=np.arange(0,1000,4)
idxpre=np.int32(nearest_n(tiempo,tpreon)[:,0])
idxpost=np.int32(nearest_n(tiempo,tposton)[:,0])
idx=np.append(idxpre,idxpost)
eeg_flat=np.ndarray.flatten(eeg_ensamblado)
EEG_f=filtro(eeg_flat,[0.1,30.0]).reshape(64,640)
PCA12(EEG_f,False)

#%% EJERCICIO 4
    
eeg_flat_bordes=np.append(np.flip(eeg_flat),eeg_flat)
eeg_flat_bordes=np.append(eeg_flat_bordes,np.flip(eeg_flat))
bandas=[[.1,4],[4,8],[8,14],[14,30]]
for i in range (len(bandas)):
    eeg_band_filt=filtro(eeg_flat_bordes,bandas[i])
    Ne=int(len(eeg_band_filt)/3)
    eeg_band_filtr=np.concatenate((eeg_band_filt[0:Ne].reshape(64,640),eeg_band_filt[Ne:2*Ne].reshape(64,640)),axis=1)
    eeg_band_filtr=np.concatenate((eeg_band_filtr,eeg_band_filt[2*Ne:3*Ne].reshape(64,640)),axis=1)
    PCA12(eeg_band_filtr,False)
    
#%% EJERCICIO 6
#SIMULACION DE ARTEFACTO DE PARPADEO
Z=3729173           #SEMILLA
N=100
idxbefore=np.arange(0,640)
mu=len(idxbefore)/2
sigma=40
latencias=(np.int16(normal(N,mu,sigma,Z)))[:-1]
desv=15
x=np.arange(0,len(idxbefore))
idxloc=np.flip(np.argsort(eeg_loc['Y'][0,:]))
geometrica=np.flip(np.sort(geometric(64,.3,Z)))
eeg_parpadeo=dataeeg
for j in range(len(idxloc)):
    for i in range(len(latencias)):
        gaussian=(1/np.sqrt(2*math.pi*desv**2))*np.exp(-(x-latencias[i])**2/(2*desv**2))*100
        eeg_parpadeo[idxloc[j],idxbefore,i]=dataeeg[idxloc[j],idxbefore,i]+(gaussian*geometrica[j])
    
eegc=np.mean(eeg_parpadeo,2)



eeg_flat=np.ndarray.flatten(eegc)
EEG_f2=filtro(eeg_flat,[0.1,30.0]).reshape(64,640)

eeg_recons=PCA12(EEG_f2,True)
plt.figure()
z=np.zeros(640)
for i in np.arange(25,26):
    for j in range(640):
        z[j]=eeg_recons[i,j]
    plt.plot(tiempo,z,label='reconstruida')
    plt.ylabel('Amplitud')
    plt.xlabel('tiempo (ms)')
    plt.title('Todos los canales')
for i in np.arange(25,26):
    plt.plot(tiempo,EEG_f[i,:],label='original')

for i in np.arange(25,26):
    plt.plot(tiempo,EEG_f2[i,:],label='artefacto')
    plt.ylabel('Amplitud')
    plt.xlabel('tiempo (ms)')
    plt.title('Todos los canales')
    
    
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('pink') 
plt.grid()
eeg_flat_bordes=np.append(np.flip(eeg_flat),eeg_flat)
eeg_flat_bordes=np.append(eeg_flat_bordes,np.flip(eeg_flat))
bandas=[[.1,4],[4,8],[8,14],[14,30]]
for i in range (len(bandas)):
    eeg_band_filt=filtro(eeg_flat_bordes,bandas[i])
    Ne=int(len(eeg_band_filt)/3)
    eeg_band_filtr=np.concatenate((eeg_band_filt[0:Ne].reshape(64,640),eeg_band_filt[Ne:2*Ne].reshape(64,640)),axis=1)
    eeg_band_filtr=np.concatenate((eeg_band_filtr,eeg_band_filt[2*Ne:3*Ne].reshape(64,640)),axis=1)
    PCA12(eeg_band_filtr,False)