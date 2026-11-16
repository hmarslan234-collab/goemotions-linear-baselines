# GoEmotions linear-baseline experiments (reproduces every number in the paper).
# Setup (run in an empty folder, then place this script beside the files):
#   B=https://raw.githubusercontent.com/google-research/google-research/master/goemotions
#   for f in train.tsv dev.tsv test.tsv emotions.txt ekman_mapping.json sentiment_mapping.json; do curl -sSO $B/data/$f; done
#   pip install numpy scipy pandas scikit-learn matplotlib   # paper used scikit-learn 1.8.0
#   python goemotions_baselines.py     # ~7 min on one CPU core; writes results.json

import json,time,sys,numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]",*a,flush=True)
emo=[l.strip() for l in open("emotions.txt")]; K=len(emo); assert K==28
def load(f):
    d=pd.read_csv(f,sep="\t",header=None,names=["text","labels","id"])
    Y=np.zeros((len(d),K),dtype=np.int8)
    for i,l in enumerate(d.labels): 
        for j in str(l).split(","): Y[i,int(j)]=1
    return d.text.tolist(),Y
Xtr,Ytr=load("train.tsv"); Xdv,Ydv=load("dev.tsv"); Xte,Yte=load("test.tsv")
log("sizes",len(Xtr),len(Xdv),len(Xte),"multi-label share train",float((Ytr.sum(1)>1).mean()))
R={"n":[len(Xtr),len(Xdv),len(Xte)],"multilabel_share":float((Ytr.sum(1)>1).mean()),"labels_per_ex":float(Ytr.sum(1).mean()),
   "emotions":emo,"train_freq":Ytr.sum(0).tolist(),"test_freq":Yte.sum(0).tolist(),"mean_tokens":float(np.mean([len(t.split()) for t in Xtr]))}

def fix(P,S):
    P=P.copy(); none=P.sum(1)==0; P[none,S[none].argmax(1)]=True; return P
def mf(Y,P,avg): return f1_score(Y,P,average=avg,zero_division=0)
def gthr(Sd,Yd):
    grid=np.quantile(Sd.ravel(),np.linspace(0.80,0.995,60)); best=(-1,None)
    for t in grid:
        s=mf(Yd,fix(Sd>=t,Sd),"macro")
        if s>best[0]: best=(s,t)
    return best[1]
def cthr(Sd,Yd):
    th=np.zeros(Sd.shape[1])
    for c in range(Sd.shape[1]):
        grid=np.quantile(Sd[:,c],np.linspace(0.5,0.998,80)); b=(-1,grid[0])
        for t in grid:
            s=f1_score(Yd[:,c],Sd[:,c]>=t,zero_division=0)
            if s>b[0]: b=(s,t)
        th[c]=b[1]
    return th
def evalS(Sd,Yd,St,Yt):
    g=gthr(Sd,Yd); c=cthr(Sd,Yd)
    Pg=fix(St>=g,St); Pc=fix(St>=c[None,:],St)
    top1=float(np.mean(Yt[np.arange(len(Yt)),St.argmax(1)]))
    return dict(dev_macro_global=mf(Yd,fix(Sd>=g,Sd),"macro"),
        test_macro_global=mf(Yt,Pg,"macro"),test_micro_global=mf(Yt,Pg,"micro"),
        test_macro_class=mf(Yt,Pc,"macro"),test_micro_class=mf(Yt,Pc,"micro"),top1=top1),Pg,Pc

def fitscore(kind,C,cw,Xa,Ya,Xbs):
    outs=[np.zeros((X.shape[0],Ya.shape[1]),dtype=np.float32) for X in Xbs]
    for c in range(Ya.shape[1]):
        y=Ya[:,c]
        if kind=="lr": m=LogisticRegression(C=C,solver="liblinear",class_weight=cw)
        elif kind=="svm": m=LinearSVC(C=C,class_weight=cw)
        elif kind=="cnb": m=ComplementNB(alpha=C)
        m.fit(Xa,y)
        for o,X in zip(outs,Xbs):
            o[:,c]= m.predict_proba(X)[:,1] if kind!="svm" else m.decision_function(X)
    return outs

# features
log("vectorizing")
vw=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,lowercase=True); Wtr=vw.fit_transform(Xtr); Wdv=vw.transform(Xdv); Wte=vw.transform(Xte)
vc=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=3,sublinear_tf=True,max_features=200000); Ctr=vc.fit_transform(Xtr); Cdv=vc.transform(Xdv); Cte=vc.transform(Xte)
Btr=sparse.hstack([Wtr,Ctr]).tocsr(); Bdv=sparse.hstack([Wdv,Cdv]).tocsr(); Bte=sparse.hstack([Wte,Cte]).tocsr()
R["nfeat"]={"word":Wtr.shape[1],"char":Ctr.shape[1]}
log("features",Wtr.shape,Ctr.shape)

res={}; store={}
# trivial baseline: always predict most frequent label (neutral)
maj=int(np.argmax(Ytr.sum(0))); P=np.zeros_like(Yte,dtype=bool); P[:,maj]=True
res["Majority label"]=dict(test_macro_global=mf(Yte,P,"macro"),test_micro_global=mf(Yte,P,"micro"),test_macro_class=None,test_micro_class=None,top1=float(Yte[:,maj].mean()),cfg="-")
R["majority_label"]=emo[maj]
def run(name,kind,feats,Cs,cw):
    Xa,Xd,Xt={"w":(Wtr,Wdv,Wte),"c":(Ctr,Cdv,Cte),"b":(Btr,Bdv,Bte)}[feats]
    best=None
    for C in Cs:
        Sd,St=fitscore(kind,C,cw,Xa,Ytr,[Xd,Xt]); m,Pg,Pc=evalS(Sd,Ydv,St,Yte)
        log(name,"C=",C,"dev macro",round(m["dev_macro_global"],4),"test macro",round(m["test_macro_global"],4))
        if best is None or m["dev_macro_global"]>best[0]["dev_macro_global"]: best=(m,C,Sd,St,Pg,Pc)
    m,C,Sd,St,Pg,Pc=best; m["cfg"]=f"{feats},C={C},cw={cw}"; res[name]=m; store[name]=(Sd,St,Pg,Pc,C); return m
run("TF-IDF word + NB","cnb","w",[0.3,1.0],None)
run("TF-IDF word + SVM","svm","w",[0.05,0.2],None)
run("TF-IDF word + SVM (balanced)","svm","w",[0.05,0.2],"balanced")
run("TF-IDF word + LR","lr","w",[2,8],None)
run("TF-IDF word + LR (balanced)","lr","w",[2,8],"balanced")
Cbest=store["TF-IDF word + LR (balanced)"][4]
run("TF-IDF char + LR (balanced)","lr","c",[Cbest],"balanced")
run("TF-IDF word+char + LR (balanced)","lr","b",[Cbest],"balanced")
R["results"]=res
json.dump(R,open("results.json","w"),indent=1)
bestname=max([k for k in res if k!="Majority label"],key=lambda k:res[k]["dev_macro_global"]); R["best"]=bestname; log("best",bestname,res[bestname]["cfg"])
Sd,St,Pg,Pc,Cb=store[bestname]
# bootstrap CIs for best model (global threshold) on test
rng=np.random.default_rng(0); n=len(Yte); bs=[]
for _ in range(200):
    idx=rng.integers(0,n,n); bs.append((mf(Yte[idx],Pg[idx],"macro"),mf(Yte[idx],Pg[idx],"micro")))
bs=np.array(bs); R["boot"]={"macro":[float(np.percentile(bs[:,0],2.5)),float(np.percentile(bs[:,0],97.5))],"micro":[float(np.percentile(bs[:,1],2.5)),float(np.percentile(bs[:,1],97.5))]}
# per-class F1 (global thr) and frequency
pcf=f1_score(Yte,Pg,average=None,zero_division=0); R["per_class_f1"]=pcf.tolist()
rho,p=spearmanr(Ytr.sum(0),pcf); R["spearman_freq_f1"]=[float(rho),float(p)]
# top confusions: for each gold label, most frequent wrongly predicted label among examples with single gold label
sing=Yte.sum(1)==1; conf=np.zeros((K,K),dtype=int)
for i in np.where(sing)[0]:
    g=int(Yte[i].argmax()); pr=np.where(Pg[i])[0]
    for q in pr:
        if q!=g: conf[g,q]+=1
pairs=sorted([(int(conf[a,b]),emo[a],emo[b]) for a in range(K) for b in range(K) if a!=b],reverse=True)[:6]; R["top_confusions"]=pairs
json.dump(R,open("results.json","w"),indent=1)
log("per-class done")
# granularity
em=json.load(open("ekman_mapping.json")); sm=json.load(open("sentiment_mapping.json"))
def grp(mp):
    names=list(mp.keys())+["neutral"]; idx=[[emo.index(e) for e in mp[k]] for k in mp]+[[emo.index("neutral")]]; return names,idx
def agg(Y,idx): return np.stack([Y[:,i].max(1) for i in idx],1).astype(np.int8)
def aggS(S,idx): return np.stack([S[:,i].max(1) for i in idx],1)
gran={}
kind,feats="lr","w"; 
for gname,mp in [("Ekman (6+neutral)",em),("Sentiment (3+neutral)",sm)]:
    names,idx=grp(mp); Yg=[agg(Y,idx) for Y in (Ytr,Ydv,Yte)]
    # (a) direct training
    Sd2,St2=fitscore("lr",Cb if False else store["TF-IDF word + LR (balanced)"][4],"balanced",Wtr,Yg[0],[Wdv,Wte]); m,_,_=evalS(Sd2,Yg[1],St2,Yg[2])
    # (b) aggregate from 28-class scores
    Sd3,St3=aggS(store["TF-IDF word + LR (balanced)"][0],idx),aggS(store["TF-IDF word + LR (balanced)"][1],idx); m3,_,_=evalS(Sd3,Yg[1],St3,Yg[2])
    gran[gname]=dict(k=len(names),direct=m["test_macro_global"],direct_micro=m["test_micro_global"],agg=m3["test_macro_global"],agg_micro=m3["test_micro_global"])
    log("gran",gname,gran[gname])
gran["Fine-grained (27+neutral)"]=dict(k=28,direct=res["TF-IDF word + LR (balanced)"]["test_macro_global"],direct_micro=res["TF-IDF word + LR (balanced)"]["test_micro_global"],agg=None,agg_micro=None)
R["granularity"]=gran; json.dump(R,open("results.json","w"),indent=1)
# learning curve (word + LR balanced), 3 subsamples per fraction
Cw=store["TF-IDF word + LR (balanced)"][4]; lc={}
for fr in [0.1,0.25,0.5]:
    vals=[]
    for s in range(3):
        r=np.random.default_rng(100+s); idx=np.sort(r.choice(len(Xtr),int(fr*len(Xtr)),replace=False))
        Sd4,St4=fitscore("lr",Cw,"balanced",Wtr[idx],Ytr[idx],[Wdv,Wte]); m,_,_=evalS(Sd4,Ydv,St4,Yte); vals.append(m["test_macro_global"])
    lc[str(fr)]=vals; log("lc",fr,vals)
lc["1.0"]=[res["TF-IDF word + LR (balanced)"]["test_macro_global"]]
R["learning_curve"]=lc; json.dump(R,open("results.json","w"),indent=1)
log("DONE")
