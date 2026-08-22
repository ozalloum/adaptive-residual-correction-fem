from __future__ import annotations
import math, time, json, platform, os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ---------------- Mesh ----------------
@dataclass
class Mesh:
    p: np.ndarray  # (N,2)
    t: np.ndarray  # (M,3), ccw

    @property
    def nnodes(self): return self.p.shape[0]
    @property
    def nelems(self): return self.t.shape[0]


def orient_ccw(p,t):
    t=t.copy()
    a=p[t[:,0]]; b=p[t[:,1]]; c=p[t[:,2]]
    det=(b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    bad=det<0
    if np.any(bad):
        tmp=t[bad,1].copy(); t[bad,1]=t[bad,2]; t[bad,2]=tmp
    return t


def compress_mesh(p,t):
    used=np.unique(t.ravel())
    newidx=-np.ones(len(p),dtype=int); newidx[used]=np.arange(len(used))
    return Mesh(p[used].copy(), orient_ccw(p[used], newidx[t]))


def make_square(n:int)->Mesh:
    xs=np.linspace(0.,1.,n+1); ys=np.linspace(0.,1.,n+1)
    P=np.array([(x,y) for y in ys for x in xs],float)
    def idx(i,j): return j*(n+1)+i
    T=[]
    for j in range(n):
        for i in range(n):
            a=idx(i,j); b=idx(i+1,j); c=idx(i+1,j+1); d=idx(i,j+1)
            if (i+j)%2==0:
                T += [[a,b,c],[a,c,d]]
            else:
                T += [[a,b,d],[b,c,d]]
    return Mesh(P, orient_ccw(P,np.asarray(T,int)))


def make_lshape(n_per_unit:int)->Mesh:
    n=2*n_per_unit
    xs=np.linspace(-1.,1.,n+1); ys=np.linspace(-1.,1.,n+1)
    P=np.array([(x,y) for y in ys for x in xs],float)
    def idx(i,j): return j*(n+1)+i
    T=[]
    for j in range(n):
        for i in range(n):
            xc=0.5*(xs[i]+xs[i+1]); yc=0.5*(ys[j]+ys[j+1])
            if xc>0 and yc<0: continue
            a=idx(i,j); b=idx(i+1,j); c=idx(i+1,j+1); d=idx(i,j+1)
            if (i+j)%2==0:
                T += [[a,b,c],[a,c,d]]
            else:
                T += [[a,b,d],[b,c,d]]
    return compress_mesh(P,np.asarray(T,int))


def edge_data(mesh:Mesh):
    e2t={}
    for k,tri in enumerate(mesh.t):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            e=(a,b) if a<b else (b,a)
            e2t.setdefault(e,[]).append(k)
    bedges=[e for e,adj in e2t.items() if len(adj)==1]
    bnodes=np.unique(np.array(bedges,dtype=int).ravel()) if bedges else np.array([],dtype=int)
    return e2t, bnodes


def refine_marked(mesh:Mesh, marked:np.ndarray)->Tuple[Mesh, Dict[int,Tuple[int,int]], np.ndarray]:
    """Adaptive longest-edge red/blue/green refinement following the
    closure used by scikit-fem MeshTri: sort each triangle so local edge
    (0,2) is longest; if either shorter edge is split, also split the longest.
    This prevents repeated green refinements from degrading mesh quality.
    """
    marked=np.asarray(marked,dtype=int)
    if len(marked)==0:
        return Mesh(mesh.p.copy(),mesh.t.copy()),{},np.arange(mesh.nnodes)
    p=mesh.p
    t=mesh.t.copy()
    V=p[t]
    l01=np.linalg.norm(V[:,0]-V[:,1],axis=1)
    l12=np.linalg.norm(V[:,1]-V[:,2],axis=1)
    l02=np.linalg.norm(V[:,0]-V[:,2],axis=1)
    ix01=(l01>l02)&(l01>l12)
    tmp=t[ix01,2].copy(); t[ix01,2]=t[ix01,1]; t[ix01,1]=tmp
    ix12=(l12>l01)&(l12>l02)
    tmp=t[ix12,0].copy(); t[ix12,0]=t[ix12,1]; t[ix12,1]=tmp

    facet_map={}; facets=[]; t2f=np.empty((len(t),3),dtype=int)
    for k,(v0,v1,v2) in enumerate(t):
        for j,(a,b) in enumerate(((v0,v1),(v1,v2),(v0,v2))):
            e=(int(a),int(b)) if a<b else (int(b),int(a))
            if e not in facet_map:
                facet_map[e]=len(facets); facets.append(e)
            t2f[k,j]=facet_map[e]
    facets=np.asarray(facets,dtype=int)
    split=np.zeros(len(facets),dtype=bool)
    split[t2f[marked].ravel()]=True
    while True:
        before=int(split.sum())
        loc=split[t2f]
        need=(loc[:,0]|loc[:,1]) & (~loc[:,2])
        if np.any(need): split[t2f[need,2]]=True
        if int(split.sum())==before: break

    split_ids=np.flatnonzero(split)
    P=p.tolist(); mid_id=-np.ones(len(facets),dtype=int); new_parents={}
    for fid in split_ids:
        a,b=facets[fid]; nid=len(P); P.append((0.5*(p[a]+p[b])).tolist())
        mid_id[fid]=nid; new_parents[nid]=(int(a),int(b))
    ix=mid_id[t2f]
    red=(ix[:,0]>=0)&(ix[:,1]>=0)&(ix[:,2]>=0)
    blue1=(ix[:,0]<0)&(ix[:,1]>=0)&(ix[:,2]>=0)
    blue2=(ix[:,0]>=0)&(ix[:,1]<0)&(ix[:,2]>=0)
    green=(ix[:,0]<0)&(ix[:,1]<0)&(ix[:,2]>=0)
    rest=(ix[:,0]<0)&(ix[:,1]<0)&(ix[:,2]<0)
    if not np.all(red|blue1|blue2|green|rest):
        raise RuntimeError('unsupported adaptive refinement pattern after closure')
    newT=[]
    for k,(v0,v1,v2) in enumerate(t):
        if rest[k]:
            newT.append([v0,v1,v2])
        elif red[k]:
            m01,m12,m02=ix[k]
            newT += [[v0,m01,m02],[v1,m01,m12],[v2,m12,m02],[m12,m02,m01]]
        elif blue1[k]:
            m12,m02=ix[k,1],ix[k,2]
            newT += [[v1,v0,m02],[v1,m12,m02],[v2,m02,m12]]
        elif blue2[k]:
            m01,m02=ix[k,0],ix[k,2]
            newT += [[v0,m01,m02],[m02,m01,v1],[v2,m02,v1]]
        elif green[k]:
            m02=ix[k,2]
            newT += [[v1,m02,v0],[v2,m02,v1]]
    P=np.asarray(P,float); T=orient_ccw(P,np.asarray(newT,int))
    return Mesh(P,T),new_parents,np.arange(mesh.nnodes)


def prolong_midpoint(old_u:np.ndarray, new_mesh:Mesh, new_parents:Dict[int,Tuple[int,int]], gfun:Callable[[np.ndarray],np.ndarray]):
    u=np.empty(new_mesh.nnodes,float); u[:len(old_u)]=old_u
    for nid,(a,b) in new_parents.items(): u[nid]=0.5*(old_u[a]+old_u[b])
    _,bn=edge_data(new_mesh); u[bn]=gfun(new_mesh.p[bn].T)
    return u


def node_adjacency(mesh:Mesh)->List[set]:
    adj=[set() for _ in range(mesh.nnodes)]
    for tri in mesh.t:
        a,b,c=map(int,tri)
        adj[a].update((b,c)); adj[b].update((a,c)); adj[c].update((a,b))
    return adj

# ---------------- Problems ----------------
@dataclass
class Problem:
    name: str
    mesh_kind: str
    D: Callable[[np.ndarray],np.ndarray]
    c: Callable[[np.ndarray],np.ndarray]
    f: Callable[[np.ndarray],np.ndarray]
    exact: Callable[[np.ndarray],np.ndarray]
    grad: Callable[[np.ndarray],np.ndarray]
    g: Callable[[np.ndarray],np.ndarray]
    notes: str=''


def problems()->Dict[str,Problem]:
    pi=np.pi
    one=lambda x: np.ones(np.asarray(x).shape[1] if np.asarray(x).ndim>1 else 1)
    zero=lambda x: np.zeros(np.asarray(x).shape[1] if np.asarray(x).ndim>1 else 1)
    def usin(x): return np.sin(pi*x[0])*np.sin(pi*x[1])
    def gusin(x): return np.vstack((pi*np.cos(pi*x[0])*np.sin(pi*x[1]),pi*np.sin(pi*x[0])*np.cos(pi*x[1])))
    smooth=Problem('Smooth Poisson','square',one,zero,lambda x:2*pi*pi*usin(x),usin,gusin,usin,'Smooth baseline; adaptivity should not outperform uniform refinement by much.')

    alpha=2/3
    def theta_l(x):
        th=np.arctan2(x[1],x[0]); return np.where(th<0,th+2*pi,th)
    def uL(x):
        r=np.sqrt(x[0]**2+x[1]**2); th=theta_l(x); return np.where(r>0,r**alpha*np.sin(alpha*th),0.0)
    def gL(x):
        r=np.sqrt(x[0]**2+x[1]**2); th=theta_l(x)
        rr=np.where(r>1e-300,r,1.0)
        ur=alpha*rr**(alpha-1)*np.sin(alpha*th)
        uth=alpha*rr**alpha*np.cos(alpha*th)
        gx=ur*np.cos(th)-uth*np.sin(th)/rr
        gy=ur*np.sin(th)+uth*np.cos(th)/rr
        gx=np.where(r>1e-14,gx,0.0); gy=np.where(r>1e-14,gy,0.0)
        return np.vstack((gx,gy))
    lshape=Problem('L-shaped singularity','lshape',one,zero,zero,uL,gL,uL,'Canonical re-entrant-corner singularity u=r^(2/3) sin(2 theta/3).')

    kappa=1e4; s=0.5
    C=((kappa-1)*s*s/2+0.5)/((kappa-1)*s+1)
    def Dint(x): return np.where(x[0] < s-1e-14,1.0,kappa)
    def gx_piece(x):
        xx=x[0]; left=xx<=s
        return np.where(left,C*xx-0.5*xx*xx,(C*xx-0.5*xx*xx)/kappa+(0.5-C)/kappa)
    def gxp_piece(x):
        xx=x[0]; return np.where(xx<=s,C-xx,(C-xx)/kappa)
    def uint(x): return gx_piece(x)*np.sin(pi*x[1])
    def guint(x): return np.vstack((gxp_piece(x)*np.sin(pi*x[1]),gx_piece(x)*pi*np.cos(pi*x[1])))
    def fint(x): return (1.0 + Dint(x)*pi*pi*gx_piece(x))*np.sin(pi*x[1])
    interface=Problem('High-contrast interface (1e4)','square',Dint,zero,fint,uint,guint,uint,'Piecewise diffusion with flux-continuous exact solution and contrast 10^4.')

    beta=120.0; x0=0.72; y0=0.33
    def ulayer(x):
        X=x[0];Y=x[1]; B=X*(1-X)*Y*(1-Y); E=np.exp(-beta*((X-x0)**2+(Y-y0)**2)); return B*E
    def glayer(x):
        X=x[0];Y=x[1]; dx=X-x0;dy=Y-y0; B=X*(1-X)*Y*(1-Y); E=np.exp(-beta*(dx*dx+dy*dy))
        Bx=(1-2*X)*Y*(1-Y); By=(1-2*Y)*X*(1-X)
        return np.vstack((E*(Bx-2*beta*dx*B), E*(By-2*beta*dy*B)))
    def flayer(x):
        X=x[0];Y=x[1]; dx=X-x0;dy=Y-y0; r2=dx*dx+dy*dy
        B=X*(1-X)*Y*(1-Y); E=np.exp(-beta*r2)
        Bx=(1-2*X)*Y*(1-Y); By=(1-2*Y)*X*(1-X)
        lapB=-2*Y*(1-Y)-2*X*(1-X)
        lapu=E*(lapB-4*beta*(dx*Bx+dy*By)+(4*beta*beta*r2-4*beta)*B)
        return -lapu + B*E
    layer=Problem('Localized reaction-diffusion layer','square',one,lambda x:np.ones(x.shape[1]),flayer,ulayer,glayer,ulayer,'Smooth but sharply localized interior feature; operator -Delta u + u.')
    return {p.name:p for p in (smooth,lshape,interface,layer)}

# ---------------- FEM ----------------
Q7_lam=np.array([
    [1/3,1/3,1/3],
    [0.470142064105115,0.470142064105115,0.059715871789770],
    [0.470142064105115,0.059715871789770,0.470142064105115],
    [0.059715871789770,0.470142064105115,0.470142064105115],
    [0.101286507323456,0.101286507323456,0.797426985353087],
    [0.101286507323456,0.797426985353087,0.101286507323456],
    [0.797426985353087,0.101286507323456,0.101286507323456],
])
Q7_w=np.array([0.225,0.132394152788506,0.132394152788506,0.132394152788506,0.125939180544827,0.125939180544827,0.125939180544827])
Q3_lam=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
Q3_w=np.array([1/3,1/3,1/3])


def geom(mesh:Mesh):
    V=mesh.p[mesh.t]
    x0,y0=V[:,0,0],V[:,0,1]; x1,y1=V[:,1,0],V[:,1,1]; x2,y2=V[:,2,0],V[:,2,1]
    det=(x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
    A=0.5*np.abs(det)
    if np.any(A<=1e-15): raise ValueError('degenerate triangle')
    G=np.empty((mesh.nelems,3,2),float)
    G[:,0,0]=(y1-y2)/det; G[:,0,1]=(x2-x1)/det
    G[:,1,0]=(y2-y0)/det; G[:,1,1]=(x0-x2)/det
    G[:,2,0]=(y0-y1)/det; G[:,2,1]=(x1-x0)/det
    return V,A,G


def eval_fun(fun, X):
    flat=X.reshape(-1,2).T; val=np.asarray(fun(flat),float)
    return val.reshape(X.shape[0],X.shape[1])


def assemble(problem:Problem, mesh:Mesh)->Tuple[csr_matrix,np.ndarray]:
    V,A,G=geom(mesh); M=mesh.nelems
    X=np.einsum('qj,mjd->mqd',Q3_lam,V)
    Dq=eval_fun(problem.D,X); cq=eval_fun(problem.c,X); fq=eval_fun(problem.f,X)
    Dint=A*np.sum(Q3_w[None,:]*Dq,axis=1)
    gd=np.einsum('mid,mjd->mij',G,G)
    K=Dint[:,None,None]*gd
    for q in range(len(Q3_w)):
        phi=Q3_lam[q]
        K += (A*Q3_w[q]*cq[:,q])[:,None,None]*(phi[None,:,None]*phi[None,None,:])
    F=np.zeros((M,3),float)
    for q in range(len(Q3_w)):
        F += (A*Q3_w[q]*fq[:,q])[:,None]*Q3_lam[q][None,:]
    tri=mesh.t
    rows=np.repeat(tri,3,axis=1).ravel(); cols=np.tile(tri,(1,3)).ravel(); vals=K.reshape(-1)
    Amat=coo_matrix((vals,(rows,cols)),shape=(mesh.nnodes,mesh.nnodes)).tocsr()
    b=np.zeros(mesh.nnodes,float); np.add.at(b,tri.ravel(),F.ravel())
    return Amat,b


def solve(problem:Problem, mesh:Mesh, A=None,b=None):
    if A is None: A,b=assemble(problem,mesh)
    _,B=edge_data(mesh); allidx=np.arange(mesh.nnodes); mask=np.ones(mesh.nnodes,bool); mask[B]=False; I=allidx[mask]
    u=np.zeros(mesh.nnodes,float); u[B]=problem.g(mesh.p[B].T)
    rhs=b[I]-A[I][:,B]@u[B]
    u[I]=spsolve(A[I][:,I],rhs)
    return u,A,b,I,B


def errors(problem:Problem, mesh:Mesh, u:np.ndarray):
    V,A,G=geom(mesh); U=u[mesh.t]; gh=np.einsum('mi,mid->md',U,G)
    l2=0.; h1s=0.; energy=0.
    for q,w in enumerate(Q7_w):
        lam=Q7_lam[q]; X=np.einsum('j,mjd->md',lam,V)
        uh=U@lam; ue=problem.exact(X.T); ge=problem.grad(X.T).T
        diff=uh-ue; gd=gh-ge
        D=problem.D(X.T); c=problem.c(X.T)
        l2 += np.sum(A*w*diff*diff)
        h1s += np.sum(A*w*np.sum(gd*gd,axis=1))
        energy += np.sum(A*w*(D*np.sum(gd*gd,axis=1)+c*diff*diff))
    return math.sqrt(l2),math.sqrt(h1s),math.sqrt(energy)


def residual_estimator(problem:Problem, mesh:Mesh, u:np.ndarray, robust_diffusion:bool=False):
    V,A,G=geom(mesh); U=u[mesh.t]; gh=np.einsum('mi,mid->md',U,G)
    E01=np.linalg.norm(V[:,1]-V[:,0],axis=1); E12=np.linalg.norm(V[:,2]-V[:,1],axis=1); E20=np.linalg.norm(V[:,0]-V[:,2],axis=1)
    h=np.maximum.reduce([E01,E12,E20]); eta2=np.zeros(mesh.nelems,float)
    for q,w in enumerate(Q3_w):
        lam=Q3_lam[q]; X=np.einsum('j,mjd->md',lam,V); uh=U@lam
        R=problem.f(X.T)-problem.c(X.T)*uh
        if robust_diffusion:
            Dloc=problem.D(X.T)
            eta2 += h*h*A*w*R*R/np.maximum(Dloc,1e-30)
        else:
            eta2 += h*h*A*w*R*R
    e2t,_=edge_data(mesh)
    cent=V.mean(axis=1); Dcent=problem.D(cent.T); flux=gh*Dcent[:,None]
    for (a,b),adj in e2t.items():
        if len(adj)!=2: continue
        k1,k2=adj; pa,pb=mesh.p[a],mesh.p[b]; ev=pb-pa; le=float(np.linalg.norm(ev)); n=np.array([ev[1],-ev[0]])/le
        jump=float(np.dot(flux[k1]-flux[k2],n))
        scale=max(Dcent[k1],Dcent[k2]) if robust_diffusion else 1.0
        cterm=0.5*le*le*jump*jump/max(scale,1e-30)
        eta2[k1]+=cterm; eta2[k2]+=cterm
    return np.sqrt(np.maximum(eta2,0.0))


def dorfler_mark(eta:np.ndarray,theta=0.5,min_mark=1):
    w=eta*eta; total=w.sum()
    if total<=0: return np.arange(min(min_mark,len(eta)))
    order=np.argsort(w)[::-1]; cum=np.cumsum(w[order]); m=np.searchsorted(cum,theta*total)+1; m=max(m,min_mark)
    return np.sort(order[:m])


def active_nodes(mesh_new:Mesh, old_n:int, rings:int):
    _,B=edge_data(mesh_new); isb=np.zeros(mesh_new.nnodes,bool); isb[B]=True
    active=set(int(i) for i in range(old_n,mesh_new.nnodes) if not isb[i])
    if rings>0:
        adj=node_adjacency(mesh_new); frontier=set(active)
        for _ in range(rings):
            nxt=set()
            for i in frontier: nxt.update(adj[i])
            nxt={j for j in nxt if not isb[j] and j not in active}
            active.update(nxt); frontier=nxt
    return np.array(sorted(active),dtype=int)


def correction_restricted(problem:Problem, mesh:Mesh, u0:np.ndarray, active:np.ndarray, A=None,b=None):
    if A is None: A,b=assemble(problem,mesh)
    r=b-A@u0
    delta=np.zeros(mesh.nnodes,float)
    if len(active): delta[active]=spsolve(A[active][:,active],r[active])
    return u0+delta, r




def correction_local_assembled(problem:Problem, mesh:Mesh, u0:np.ndarray, active:np.ndarray):
    """Solve the residual correction using element assembly restricted to active rows/cols.

    This forms neither the global stiffness matrix nor the global load vector.  Only
    elements incident to at least one active degree of freedom are integrated.
    The returned solution is algebraically identical (up to roundoff) to restricting
    the globally assembled Galerkin system to the same active subspace.
    """
    active=np.asarray(active,dtype=int)
    if len(active)==0:
        return u0.copy(), 0, 0
    g2l=np.full(mesh.nnodes,-1,dtype=int); g2l[active]=np.arange(len(active))
    amask=np.zeros(mesh.nnodes,dtype=bool); amask[active]=True
    touched=np.where(np.any(amask[mesh.t],axis=1))[0]
    tri=mesh.t[touched]; V=mesh.p[tri]
    x0,y0=V[:,0,0],V[:,0,1]; x1,y1=V[:,1,0],V[:,1,1]; x2,y2=V[:,2,0],V[:,2,1]
    det=(x1-x0)*(y2-y0)-(y1-y0)*(x2-x0); area=0.5*np.abs(det)
    G=np.empty((len(tri),3,2),float)
    G[:,0,0]=(y1-y2)/det; G[:,0,1]=(x2-x1)/det
    G[:,1,0]=(y2-y0)/det; G[:,1,1]=(x0-x2)/det
    G[:,2,0]=(y0-y1)/det; G[:,2,1]=(x1-x0)/det
    X=np.einsum('qj,mjd->mqd',Q3_lam,V)
    Dq=eval_fun(problem.D,X); cq=eval_fun(problem.c,X); fq=eval_fun(problem.f,X)
    Dint=area*np.sum(Q3_w[None,:]*Dq,axis=1)
    gd=np.einsum('mid,mjd->mij',G,G)
    K=Dint[:,None,None]*gd
    for q in range(len(Q3_w)):
        phi=Q3_lam[q]
        K += (area*Q3_w[q]*cq[:,q])[:,None,None]*(phi[None,:,None]*phi[None,None,:])
    F=np.zeros((len(tri),3),float)
    for q in range(len(Q3_w)):
        F += (area*Q3_w[q]*fq[:,q])[:,None]*Q3_lam[q][None,:]
    rhs=np.zeros(len(active),float); rr=[]; cc=[]; vv=[]
    for e,nodes in enumerate(tri):
        uloc=u0[nodes]
        for a,ga in enumerate(nodes):
            ia=g2l[ga]
            if ia<0: continue
            rhs[ia] += F[e,a] - float(K[e,a,:]@uloc)
            for bj,gb in enumerate(nodes):
                jb=g2l[gb]
                if jb>=0:
                    rr.append(ia); cc.append(jb); vv.append(K[e,a,bj])
    Kaa=coo_matrix((vv,(rr,cc)),shape=(len(active),len(active))).tocsr()
    delta=spsolve(Kaa,rhs)
    u=u0.copy(); u[active]+=delta
    return u, len(touched), Kaa.nnz



def residual_vector_diag(problem:Problem, mesh:Mesh, u:np.ndarray):
    """Assemble r=b-Au and diag(A) without forming the global sparse matrix."""
    V,Ae,G=geom(mesh); M=mesh.nelems
    X=np.einsum('qj,mjd->mqd',Q3_lam,V)
    Dq=eval_fun(problem.D,X); cq=eval_fun(problem.c,X); fq=eval_fun(problem.f,X)
    Dint=Ae*np.sum(Q3_w[None,:]*Dq,axis=1)
    gd=np.einsum('mid,mjd->mij',G,G)
    K=Dint[:,None,None]*gd
    for q in range(len(Q3_w)):
        phi=Q3_lam[q]
        K += (Ae*Q3_w[q]*cq[:,q])[:,None,None]*(phi[None,:,None]*phi[None,None,:])
    F=np.zeros((M,3),float)
    for q in range(len(Q3_w)):
        F += (Ae*Q3_w[q]*fq[:,q])[:,None]*Q3_lam[q][None,:]
    U=u[mesh.t]
    Rloc=F-np.einsum('mij,mj->mi',K,U)
    r=np.zeros(mesh.nnodes,float); d=np.zeros(mesh.nnodes,float)
    np.add.at(r,mesh.t.ravel(),Rloc.ravel())
    for a in range(3): np.add.at(d,mesh.t[:,a],K[:,a,a])
    return r,d

def scaled_residual_matrixfree(problem:Problem,mesh:Mesh,u:np.ndarray,I:np.ndarray):
    r,d=residual_vector_diag(problem,mesh,u)
    good=np.maximum(np.abs(d[I]),1e-30)
    return float(np.linalg.norm(r[I]/np.sqrt(good)))

def scaled_residual(A,b,u,I):
    r=b-A@u; d=A.diagonal(); good=np.maximum(np.abs(d[I]),1e-30)
    return float(np.linalg.norm(r[I]/np.sqrt(good)))

# ---------------- Experiment runners ----------------
def fixed_uniform(problem,n):
    mesh=make_square(n) if problem.mesh_kind=='square' else make_lshape(n)
    t0=time.perf_counter(); A,b=assemble(problem,mesh); t1=time.perf_counter(); u,A,b,I,B=solve(problem,mesh,A,b); t2=time.perf_counter()
    e=errors(problem,mesh,u); eta=residual_estimator(problem,mesh,u,robust_diffusion=('High-contrast' in problem.name))
    return mesh,u,A,b,I,B,e,eta,(t1-t0,t2-t1)


def afem_sequence(problem,n0=8,levels=7,theta=0.5):
    mesh=make_square(n0) if problem.mesh_kind=='square' else make_lshape(n0)
    rows=[]; snapshots=[]; cumulative=0.0
    for lev in range(levels+1):
        t0=time.perf_counter(); A,b=assemble(problem,mesh); u,A,b,I,B=solve(problem,mesh,A,b); t1=time.perf_counter(); eta=residual_estimator(problem,mesh,u,robust_diffusion=('High-contrast' in problem.name)); t2=time.perf_counter()
        l2,h1,en=errors(problem,mesh,u); cumulative += (t2-t0)
        rows.append(dict(Problem=problem.name,Method='AFEM-direct',Level=lev,DOF=mesh.nnodes,Cells=mesh.nelems,L2=l2,H1semi=h1,Energy=en,Estimator=float(np.linalg.norm(eta)),Cycle_s=t2-t0,Cumulative_s=cumulative))
        snapshots.append((mesh,u,A,b,I,B,eta.copy()))
        if lev<levels:
            marked=dorfler_mark(eta,theta); tr=time.perf_counter(); mesh,_,_=refine_marked(mesh,marked); cumulative += time.perf_counter()-tr
    return pd.DataFrame(rows),snapshots


def local_correction_sequence(problem,n0=8,levels=7,theta=0.5,rings=1,auto_tau:Optional[float]=None,max_rings=4):
    mesh=make_square(n0) if problem.mesh_kind=='square' else make_lshape(n0)
    # initial direct solve; subsequent corrections use local element assembly only
    t0=time.perf_counter(); A,b=assemble(problem,mesh); u,A,b,I,B=solve(problem,mesh,A,b); eta=residual_estimator(problem,mesh,u,robust_diffusion=('High-contrast' in problem.name)); t1=time.perf_counter(); cumulative=t1-t0
    l2,h1,en=errors(problem,mesh,u)
    rows=[dict(Problem=problem.name,Method=('RCHC' if auto_tau is not None else f'HLRC-r{rings}'),Level=0,DOF=mesh.nnodes,Cells=mesh.nelems,ActiveDOF=len(I),ActiveFraction=1.0,Halo=rings if auto_tau is None else -1,TouchedCells=mesh.nelems,LocalNNZ=A[I][:,I].nnz,L2=l2,H1semi=h1,Energy=en,Estimator=float(np.linalg.norm(eta)),ResidualReduction=0.0,Cycle_s=t1-t0,Cumulative_s=cumulative)]
    for lev in range(1,levels+1):
        marked=dorfler_mark(eta,theta); oldmesh=mesh; oldu=u; tr=time.perf_counter(); mesh,parents,_=refine_marked(oldmesh,marked); tref=time.perf_counter()-tr
        u0=prolong_midpoint(oldu,mesh,parents,problem.g)
        _,B=edge_data(mesh); mask=np.ones(mesh.nnodes,bool);mask[B]=False; I=np.arange(mesh.nnodes)[mask]
        ta=time.perf_counter(); r0=scaled_residual_matrixfree(problem,mesh,u0,I)
        used_rings=rings; ntouch=0; lnnz=0
        if auto_tau is None:
            act=active_nodes(mesh,oldmesh.nnodes,rings); u,ntouch,lnnz=correction_local_assembled(problem,mesh,u0,act)
            rr=scaled_residual_matrixfree(problem,mesh,u,I)/(r0 if r0>0 else 1.0)
        else:
            u=u0.copy(); act=np.array([],dtype=int); rr=1.0
            for kr in range(max_rings+1):
                act=active_nodes(mesh,oldmesh.nnodes,kr)
                u,ntouch,lnnz=correction_local_assembled(problem,mesh,u0,act)
                rr=scaled_residual_matrixfree(problem,mesh,u,I)/(r0 if r0>0 else 1.0)
                used_rings=kr
                if rr<=auto_tau: break
        eta=residual_estimator(problem,mesh,u,robust_diffusion=('High-contrast' in problem.name)); tb=time.perf_counter(); cumulative += tref+(tb-ta)
        l2,h1,en=errors(problem,mesh,u)
        rows.append(dict(Problem=problem.name,Method=('RCHC' if auto_tau is not None else f'HLRC-r{rings}'),Level=lev,DOF=mesh.nnodes,Cells=mesh.nelems,ActiveDOF=len(act),ActiveFraction=len(act)/max(len(I),1),Halo=used_rings,TouchedCells=ntouch,LocalNNZ=lnnz,L2=l2,H1semi=h1,Energy=en,Estimator=float(np.linalg.norm(eta)),ResidualReduction=rr,Cycle_s=tref+(tb-ta),Cumulative_s=cumulative))
    return pd.DataFrame(rows)

def locality_spectrum(problem,afem_snapshots,max_level=None,rings_list=(0,1,2,3)):
    rows=[]
    L=len(afem_snapshots)-1 if max_level is None else min(max_level,len(afem_snapshots)-1)
    for lev in range(1,L+1):
        oldmesh,oldu,*_=afem_snapshots[lev-1]
        mesh,ud,A,b,I,B,eta=afem_snapshots[lev]
        oldeta=afem_snapshots[lev-1][-1]; marked=dorfler_mark(oldeta,0.5); remesh,parents,_=refine_marked(oldmesh,marked)
        assert remesh.nnodes==mesh.nnodes and remesh.nelems==mesh.nelems and np.allclose(remesh.p,mesh.p)
        u0=prolong_midpoint(oldu,mesh,parents,problem.g)
        base_e=errors(problem,mesh,u0)[2]
        direct_e=errors(problem,mesh,ud)[2]
        corr_total=max(base_e*base_e-direct_e*direct_e,1e-30)
        for kr in rings_list:
            act=active_nodes(mesh,oldmesh.nnodes,kr)
            t0=time.perf_counter(); ul,ntouch,lnnz=correction_local_assembled(problem,mesh,u0,act); t1=time.perf_counter()
            en=errors(problem,mesh,ul)[2]
            recovered=max(base_e*base_e-en*en,0.0)
            rho=min(max(recovered/corr_total,0.0),1.0+1e-8)
            rows.append(dict(Problem=problem.name,Level=lev,Halo=kr,DOF=mesh.nnodes,InteriorDOF=len(I),ActiveDOF=len(act),ActiveFraction=len(act)/len(I),TouchedCells=ntouch,LocalNNZ=lnnz,Energy=en,DirectEnergy=direct_e,ProlongedEnergy=base_e,RecoveryFraction=rho,WorkNormalizedRecovery=rho/(len(act)/len(I)) if len(act)>0 else 0.0,Solve_s=t1-t0))
        t0=time.perf_counter(); ug,ntouch,lnnz=correction_local_assembled(problem,mesh,u0,I); t1=time.perf_counter(); en=errors(problem,mesh,ug)[2]
        rows.append(dict(Problem=problem.name,Level=lev,Halo=999,DOF=mesh.nnodes,InteriorDOF=len(I),ActiveDOF=len(I),ActiveFraction=1.0,TouchedCells=ntouch,LocalNNZ=lnnz,Energy=en,DirectEnergy=direct_e,ProlongedEnergy=base_e,RecoveryFraction=1.0,WorkNormalizedRecovery=1.0,Solve_s=t1-t0))
    return pd.DataFrame(rows)


def timed_solve(problem,mesh,mode='direct',u0=None,active=None,repeats=25):
    vals=[]
    for _ in range(3):
        A,b=assemble(problem,mesh)
        if mode=='direct': solve(problem,mesh,A,b)
        else: correction_restricted(problem,mesh,u0,active,A,b)
    for _ in range(repeats):
        t0=time.perf_counter(); A,b=assemble(problem,mesh)
        if mode=='direct': solve(problem,mesh,A,b)
        else: correction_restricted(problem,mesh,u0,active,A,b)
        vals.append(time.perf_counter()-t0)
    a=np.array(vals)
    return dict(n=repeats,median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),mean=float(a.mean()),sd=float(a.std(ddof=1)))


def main(outdir=None):
    if outdir is None:
        outdir = str(Path(__file__).resolve().parents[1] / 'regenerated_primary')
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    probs=problems(); all_afem=[]; all_local=[]; all_spec=[]; all_uniform=[]; snaps={}
    for pname,p in probs.items():
        print('\n###',pname,flush=True)
        ns=[8,12,16,24,32,48,64] if p.mesh_kind=='square' else [4,6,8,12,16,24,32]
        for n in ns:
            mesh,u,A,b,I,B,e,eta,tms=fixed_uniform(p,n)
            print('uniform',n,mesh.nnodes,e,flush=True)
            all_uniform.append(dict(Problem=pname,Method='Uniform',n=n,DOF=mesh.nnodes,Cells=mesh.nelems,L2=e[0],H1semi=e[1],Energy=e[2],Estimator=float(np.linalg.norm(eta)),Assembly_s=tms[0],Solve_s=tms[1]))
        df,sn=afem_sequence(p,n0=(8 if p.mesh_kind=='square' else 4),levels=7,theta=.5); all_afem.append(df);snaps[pname]=sn
        print(df[['Level','DOF','Energy','Estimator']].to_string(index=False),flush=True)
        for r in (0,1,2): all_local.append(local_correction_sequence(p,n0=(8 if p.mesh_kind=='square' else 4),levels=7,theta=.5,rings=r))
        all_local.append(local_correction_sequence(p,n0=(8 if p.mesh_kind=='square' else 4),levels=7,theta=.5,auto_tau=.15,max_rings=4))
        all_spec.append(locality_spectrum(p,sn,max_level=7,rings_list=(0,1,2,3)))
    uni=pd.DataFrame(all_uniform); af=pd.concat(all_afem,ignore_index=True); loc=pd.concat(all_local,ignore_index=True); spec=pd.concat(all_spec,ignore_index=True)
    uni.to_csv(out/'uniform.csv',index=False); af.to_csv(out/'afem.csv',index=False); loc.to_csv(out/'localized_correction.csv',index=False); spec.to_csv(out/'locality_spectrum.csv',index=False)
    env=dict(date=time.strftime('%Y-%m-%d'),python=platform.python_version(),platform=platform.platform(),cpu=os.cpu_count(),numpy=np.__version__,scipy=__import__('scipy').__version__,timing_repeats=25)
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    print('Wrote',out)

if __name__=='__main__': main()
