"""Five-frequency probe, bounded NLS and causal weighted extrapolation."""
import csv
import json
from pathlib import Path

import numpy as np


FS = np.array([59000, 64000, 69000, 74000, 79000])
LOW = np.log([.7, .8, .8])
HIGH = np.log([1.3, 1.2, 1.2])


def impedance(row):
    return np.array([[complex(row[s + "_re_ohm"], row[s + "_im_ohm"])
                      for s in ("Z11", "Z12")],
                     [complex(row[s + "_re_ohm"], row[s + "_im_ohm"])
                      for s in ("Z21", "Z22")]], dtype=complex)


def design_circuit(l1, l2, ratio, cfg):
    w = 2*np.pi*cfg["reference_frequency_Hz"]
    return {"Lf": ratio*l1, "Cf": 1/(w*w*ratio*l1),
            "Cp": 1/(w*w*(1-ratio)*l1), "Cs": 1/(w*w*l2),
            "Vs": cfg["source_Vrms"], "Rs": cfg["source_resistance_ohm"]}


class CurrentEstimator:
    """The nominal two-port model at zero lateral offset and 4 mm surface gap."""

    def __init__(self, raw_impedance, circuit_configuration):
        with Path(raw_impedance).open(encoding="utf-8-sig", newline="") as stream:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        nominal = [row for row in rows if row["x_m"] == row["y_m"] == 0 and row["z_m"] == -.005]
        by_freq = {row["freq_Hz"]: impedance(row) for row in nominal}
        if len(nominal) != len(FS) or set(by_freq) != set(FS):
            raise ValueError("Expected one nominal two-port at each of the five probe frequencies")
        self.nom = np.stack([by_freq[f] for f in FS])
        z = self.nom[2]
        self.ref = np.array([(z[0, 1].imag + z[1, 0].imag)/2, z[0, 0].imag, z[1, 1].imag]) / (2*np.pi*69000)
        config = json.loads(Path(circuit_configuration).read_text(encoding="utf-8-sig"))
        self.c = design_circuit(self.ref[1], self.ref[2], .4, config)
        self.scale = abs(self.probe(self.nom))

    def decode(self,theta):
        q=np.exp(theta); a=np.broadcast_to(self.nom,q.shape[:-1]+self.nom.shape).copy()
        for i,j,k in [(0,0,1),(1,1,2),(0,1,0),(1,0,0)]:
            a[...,i,j]=self.nom[:,i,j].real+1j*self.nom[:,i,j].imag*q[...,k,None]
        return a

    def probe(self,z):
        w=2*np.pi*FS;c=self.c
        den=z[...,1,1]+1/(1j*w*c['Cs'])+10
        branch=1/(1j*w*c['Cp'])+z[...,0,0]-z[...,0,1]*z[...,1,0]/den
        zin=c['Rs']+1j*w*c['Lf']+1/(1/branch+1j*w*c['Cf'])
        return c['Vs']/zin

    def estimate(self,obs):
        """Same bounded, multi-start, known-RL NLS for all learned/linear methods."""
        target=np.concatenate([(obs/self.scale).real,(obs/self.scale).imag],axis=-1)
        n=len(obs); best=np.zeros((n,3));bc=np.full(n,np.inf)
        def feat(x):
            y=self.probe(self.decode(x))/self.scale
            return np.concatenate([y.real,y.imag],axis=-1)
        for start in [np.zeros(3),np.log([.9,1,.9]),np.log([1.1,1,1.1])]:
            x=np.broadcast_to(start,(n,3)).copy(); damp=np.full(n,1e-3)
            for it in range(28):
                p=feat(x);r=p-target
                eps=1e-5
                j=np.stack([(feat(x+eps*np.eye(3)[k])-p)/eps for k in range(3)],axis=-1)
                jt=np.swapaxes(j,-1,-2); h=jt@j;g=np.einsum('nki,ni->nk',jt,r)
                diag=np.maximum(np.diagonal(h,axis1=1,axis2=2),1e-7)
                step=np.linalg.solve(h+damp[:,None,None]*np.eye(3)[None,:,:]*diag[:,None,:],g[...,None])[...,0]
                nx=np.clip(x-step,LOW,HIGH); nr=feat(nx)-target
                accept=(nr*nr).sum(1)<(r*r).sum(1)
                x[accept]=nx[accept];damp=np.where(accept,np.maximum(damp/3,1e-10),np.minimum(damp*5,1e8))
            cost=((feat(x)-target)**2).sum(1);take=cost<bc
            bc[take]=cost[take];best[take]=x[take]
        assert np.isfinite(best).all()
        return best


def linear_prediction(est,win,decay):
    n=len(est);ids=np.arange(n)[:,None]-np.arange(win-1,-1,-1)[None,:]
    values=est[np.maximum(ids,0)]
    x=np.arange(-win+1,1,dtype=float);w=np.exp(decay*x);w/=w.sum()
    xm=np.sum(w*x);den=np.sum(w*(x-xm)**2)
    slope=np.einsum('w,nwc->nc',w*(x-xm)/den,values)
    level=np.einsum('w,nwc->nc',w,values)-xm*slope
    hs=np.arange(1,5)
    return level,level[:,None,:]+slope[:,None,:]*hs[None,:,None]
