from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class DistanceModel(nn.Module):
    higher_is_better=False
    def loss(self,pos,neg): return F.softplus(pos[:,None]-neg+getattr(self,'margin',6.)).mean()

class TransE(DistanceModel):
    def __init__(self,nentity,nrelation,dim=200,margin=6.):
        super().__init__(); self.margin=margin; self.e=nn.Embedding(nentity,dim); self.r=nn.Embedding(nrelation,dim)
        nn.init.xavier_uniform_(self.e.weight); nn.init.xavier_uniform_(self.r.weight)
    def forward(self,h,r,t): return torch.linalg.vector_norm(self.e(h)+self.r(r)-self.e(t),ord=1,dim=-1)

class TransERR(DistanceModel):
    def __init__(self,nentity,nrelation,dim=200,margin=6.):
        super().__init__(); self.margin=margin; self.e=nn.Embedding(nentity,dim); self.direction=nn.Embedding(nrelation,dim); self.bias=nn.Embedding(nrelation,dim)
        for x in (self.e,self.direction,self.bias): nn.init.xavier_uniform_(x.weight)
    def forward(self,h,r,t): return torch.linalg.vector_norm(self.e(h)*F.normalize(self.direction(r),dim=-1)+self.bias(r)-self.e(t),ord=1,dim=-1)

class RotatE(DistanceModel):
    def __init__(self,nentity,nrelation,dim=200,margin=6.):
        super().__init__(); self.margin=margin; self.er=nn.Embedding(nentity,dim); self.ei=nn.Embedding(nentity,dim); self.phase=nn.Embedding(nrelation,dim)
        nn.init.xavier_uniform_(self.er.weight); nn.init.xavier_uniform_(self.ei.weight); nn.init.uniform_(self.phase.weight,0,2*math.pi)
    def forward(self,h,r,t):
        rr,ri=torch.cos(self.phase(r)),torch.sin(self.phase(r)); hr,hi=self.er(h),self.ei(h)
        return torch.linalg.vector_norm(torch.cat((hr*rr-hi*ri-self.er(t),hr*ri+hi*rr-self.ei(t)),-1),ord=1,dim=-1)

def qmul(a,b):
    ar,ai,aj,ak=a; br,bi,bj,bk=b
    return ar*br-ai*bi-aj*bj-ak*bk, ar*bi+ai*br+aj*bk-ak*bj, ar*bj-ai*bk+aj*br+ak*bi, ar*bk+ai*bj-aj*bi+ak*br

class QuatE(nn.Module):
    higher_is_better=True
    def __init__(self,nentity,nrelation,dim=200,margin=6.):
        super().__init__(); self.margin=margin; self.e=nn.ModuleList([nn.Embedding(nentity,dim) for _ in range(4)]); self.r=nn.ModuleList([nn.Embedding(nrelation,dim) for _ in range(4)])
        for x in list(self.e)+list(self.r): nn.init.xavier_uniform_(x.weight)
    def forward(self,h,r,t):
        rv=[x(r) for x in self.r]; norm=torch.sqrt(sum(x.square() for x in rv)+1e-12); rv=[x/norm for x in rv]
        z=qmul([x(h) for x in self.e],rv); return sum(x*y(t) for x,y in zip(z,self.e)).sum(-1)
    def loss(self,pos,neg): return (F.softplus(-pos)+F.softplus(neg).mean(1)).mean()

class QuatED(DistanceModel):
    def __init__(self,nentity,nrelation,dim=200,margin=6.):
        super().__init__(); self.margin=margin; self.e=nn.ModuleList([nn.Embedding(nentity,dim) for _ in range(4)]); self.r=nn.ModuleList([nn.Embedding(nrelation,dim) for _ in range(4)]); self.mod=nn.Embedding(nrelation,1)
        for x in list(self.e)+list(self.r): nn.init.xavier_uniform_(x.weight)
        nn.init.ones_(self.mod.weight)
    def forward(self,h,r,t):
        rv=[x(r) for x in self.r]; norm=torch.sqrt(sum(x.square() for x in rv)+1e-12); z=qmul([x(h) for x in self.e],[x/norm for x in rv]); m=self.mod(r)
        return torch.linalg.vector_norm(torch.cat([x*m-y(t) for x,y in zip(z,self.e)],-1),ord=1,dim=-1)


class LLMGeoSE(DistanceModel):
    """Quaternion rotation with a relation-conditioned dual quaternion.

    The four-dimensional quaternion structure is retained.  A single shared
    linear layer maps the relation representation to the four components of
    the dual (translation/bias) quaternion, as required by LLM-GeoSE.
    """
    def __init__(self,nentity,nrelation,dim=200,margin=6.):
        super().__init__(); self.margin=margin; self.dim=dim
        self.e=nn.ModuleList([nn.Embedding(nentity,dim) for _ in range(4)])
        self.r=nn.ModuleList([nn.Embedding(nrelation,dim) for _ in range(4)])
        self.rel_dual_linear=nn.Linear(dim*4,dim*4)
        for x in list(self.e)+list(self.r): nn.init.xavier_uniform_(x.weight)
        nn.init.xavier_uniform_(self.rel_dual_linear.weight); nn.init.zeros_(self.rel_dual_linear.bias)
    def forward(self,h,r,t):
        raw=[x(r) for x in self.r]
        norm=torch.sqrt(sum(x.square() for x in raw)+1e-12)
        rotation=[x/norm for x in raw]
        dual=self.rel_dual_linear(torch.cat(raw,-1)).chunk(4,dim=-1)
        transformed=qmul([x(h) for x in self.e],rotation)
        return torch.linalg.vector_norm(
            torch.cat([x+d-y(t) for x,d,y in zip(transformed,dual,self.e)],-1),ord=1,dim=-1)


class RoCS(nn.Module):
    """Wang et al. (2024): joint cosine of h∘r and t; larger is better."""
    higher_is_better = True
    def __init__(self, nentity, nrelation, dim=200, gamma=12., mu=20., alpha=1.):
        super().__init__(); self.gamma=gamma; self.mu=mu; self.alpha=alpha
        self.er=nn.Embedding(nentity,dim); self.ei=nn.Embedding(nentity,dim)
        self.rr=nn.Embedding(nrelation,dim); self.ri=nn.Embedding(nrelation,dim)
        for e in (self.er,self.ei,self.rr,self.ri): nn.init.xavier_uniform_(e.weight)
    def forward(self,h,r,t):
        hr,hi=self.er(h),self.ei(h); rr,ri=self.rr(r),self.ri(r)
        norm=(rr.square()+ri.square()+1e-12).sqrt(); rr,ri=rr/norm,ri/norm
        xr=hr*rr-hi*ri; xi=hr*ri+hi*rr
        x=torch.cat((xr,xi),-1); y=torch.cat((self.er(t),self.ei(t)),-1)
        return F.cosine_similarity(x,y,dim=-1,eps=1e-12)
    def loss(self,pos,neg):
        weights=F.softmax(self.alpha*neg,dim=1).detach()
        return (-F.logsigmoid(self.mu*pos-self.gamma) - (weights*F.logsigmoid(self.gamma-self.mu*neg)).sum(1)).mean()


class HarmonicTimeEncoder(nn.Module):
    """Torch adaptation of hfgeo/TimeEmbedding TheoryTimeRelationEncoder."""
    def __init__(self, out_dim):
        super().__init__()
        unit=torch.tensor([1.,1/12,1/360,1/8640,1/518400,1/31104000])
        self.register_buffer("unit",unit); self.single=nn.Sequential(nn.Linear(6,out_dim),nn.LeakyReLU())
        self.out=nn.Linear(out_dim*8,out_dim)
    def forward(self,x):
        a=x*self.unit; b=-x*self.unit
        m1,m2=self.single(x),self.single(x)
        parts=(m1,m2,self.single(a),self.single(b),self.single(torch.sin(a)),self.single(torch.sin(b)),m2-m1,(m1+m2)/2)
        return self.out(torch.cat(parts,-1))


class TFKGE(nn.Module):
    """Link-prediction adapter using the authors' full harmonic temporal encoder."""
    higher_is_better=False
    def __init__(self,nentity,nrelation,time_matrix,dim=200,margin=6.):
        super().__init__(); self.margin=margin
        self.e=nn.Embedding(nentity,dim); self.r=nn.Embedding(nrelation,dim)
        self.time=HarmonicTimeEncoder(dim); self.register_buffer("time_matrix",time_matrix)
        nn.init.xavier_uniform_(self.e.weight); nn.init.xavier_uniform_(self.r.weight)
    def forward(self,h,r,t):
        return torch.linalg.vector_norm(self.e(h)+self.r(r)+self.time(self.time_matrix[h])-self.e(t),ord=1,dim=-1)
    def loss(self,pos,neg): return F.softplus(pos[:,None]-neg+self.margin).mean()


class QiuSTCM(nn.Module):
    """Qiu et al. (2025) inspired link-prediction adapter, not the paper's original model.

    It combines structural score with entity type, time and spatial features. Optional
    rule/dependency scores may be supplied by a downstream corpus pipeline.
    """
    higher_is_better=False
    def __init__(self,nentity,nrelation,time_matrix,coord_matrix,dim=200,margin=6.):
        super().__init__(); self.margin=margin
        self.e=nn.Embedding(nentity,dim); self.r=nn.Embedding(nrelation,dim)
        self.time=HarmonicTimeEncoder(dim); self.space=nn.Sequential(nn.Linear(4,dim),nn.Tanh())
        self.gate=nn.Embedding(nrelation,3); self.register_buffer("time_matrix",time_matrix); self.register_buffer("coord_matrix",coord_matrix)
        nn.init.xavier_uniform_(self.e.weight); nn.init.xavier_uniform_(self.r.weight); nn.init.zeros_(self.gate.weight)
    def _space_features(self,ids):
        c=self.coord_matrix[ids]; lat=torch.deg2rad(c[:,0]); lon=torch.deg2rad(c[:,1])
        return torch.stack((torch.sin(lat),torch.cos(lat),torch.sin(lon),torch.cos(lon)),-1)
    def forward(self,h,r,t):
        g=torch.sigmoid(self.gate(r)); base=self.e(h)+self.r(r)
        lhs=base+g[:,0:1]*self.time(self.time_matrix[h])+g[:,1:2]*self.space(self._space_features(h))
        rhs=self.e(t)+g[:,2:3]*self.space(self._space_features(t))
        return torch.linalg.vector_norm(lhs-rhs,ord=1,dim=-1)
    def loss(self,pos,neg): return F.softplus(pos[:,None]-neg+self.margin).mean()
