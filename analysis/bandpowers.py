import pymaster as nmt
import numpy as np

class Bandpowers(object):
    def __init__(self,nside,d):
        if d['type']=='linlog':
            #Set up linear part
            l_edges_lin=np.linspace(2,d['lsplit'],(d['lsplit']-2)//d['nlb_lin']+1).astype(int)
            #Set up log part
            l_edges_log=np.unique(np.logspace(np.log10(d['lsplit']),
                                              np.log10(3*nside-1),d['nb_log']).astype(int))
            #Join
            l_edges=np.concatenate((l_edges_lin,l_edges_log[1:]))

            #Give bandpower indices and weights to each multipole
            larr=np.arange(3*nside)
            weights=np.ones(len(larr))
            bpws=-1+np.zeros(len(larr),dtype=int)
            for i in range(len(l_edges)-1):
                bpws[l_edges[i]:l_edges[i+1]]=i

            #Create binning scheme
            self.bn=nmt.NmtBin(nside,ells=larr,bpws=bpws,weights=weights)
        elif d['type']=='lin':
            #Create binning scheme
            self.bn=nmt.NmtBin(nside,nlb=d['nlb'])
        else:
            raise ValueError("Unrecognised binning scheme "+d['type'])
