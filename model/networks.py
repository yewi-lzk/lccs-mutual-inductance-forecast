"""Frozen four-step and six-step network architectures."""
import torch
from torch import nn
from torch.nn import functional as F


STEPS = 6


class Block(nn.Module):
    def __init__(self,cin,cout,d):
        super().__init__();self.d=d
        self.c1=nn.Conv1d(cin,cout,3,dilation=d);self.c2=nn.Conv1d(cout,cout,3,dilation=d)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
        self.drop=nn.Dropout(.05)
    def forward(self,x):
        a=self.drop(F.relu(self.c1(F.pad(x,(2*self.d,0)))))
        a=self.drop(F.relu(self.c2(F.pad(a,(2*self.d,0)))))
        return F.relu(a+self.skip(x))


class Forecast(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name
        if name=='GRU': self.gru=nn.GRU(3,24,batch_first=True);nin=24
        elif name=='Transformer':
            self.embed=nn.Linear(3,16);self.pos=nn.Parameter(torch.zeros(1,32,16))
            layer=nn.TransformerEncoderLayer(16,2,32,.05,batch_first=True)
            self.transformer=nn.TransformerEncoder(layer,1,enable_nested_tensor=False);nin=16
        else:
            self.blocks=nn.Sequential(Block(3,16,1),Block(16,16,2),Block(16,16,4));nin=16
            if 'Attention' in name:self.query=nn.Linear(16,16,bias=False)
        self.head=nn.Linear(nin,12)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        if self.name=='GRU': a,_=self.gru(x.transpose(1,2));a=a[:,-1]
        elif self.name=='Transformer':
            a=self.embed(x.transpose(1,2))+self.pos
            mask=torch.triu(torch.ones(32,32,device=x.device,dtype=torch.bool),diagonal=1)
            a=self.transformer(a,mask=mask)[:,-1]
        else:
            h=self.blocks(x);a=h[:,:,-1]
            if 'Attention' in self.name:
                q=self.query(a);weights=torch.softmax(torch.einsum('bc,bct->bt',q,h)/4,dim=-1)
                a=torch.einsum('bt,bct->bc',weights,h)
        return self.head(a).reshape(-1,4,3)


class Forecast30(nn.Module):
    """Same causal encoder and zero-initialized output layer as the four-step fit."""
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "GRU":
            self.gru = nn.GRU(3, 24, batch_first=True)
            channels = 24
        elif name == "TCN_Residual":
            self.blocks = nn.Sequential(Block(3, 16, 1), Block(16, 16, 2), Block(16, 16, 4))
            channels = 16
        else:
            raise ValueError(name)
        self.head = nn.Linear(channels, STEPS * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if self.name == "GRU":
            features, _ = self.gru(x.transpose(1, 2))
            last = features[:, -1]
        else:
            last = self.blocks(x)[:, :, -1]
        return self.head(last).reshape(-1, STEPS, 3)
