# Sanity controls: shuffled-label run, manual F1 recomputation, split overlap, default-threshold run.
# Needs the data files and goemotions_baselines.py renamed/copied as run.py in the same folder.

import numpy as np,re,sys
src=open("run.py").read().split("# features")[0]   # reuse loading + metric helpers
exec(src)
from sklearn.feature_extraction.text import TfidfVectorizer
vw=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True); Wtr=vw.fit_transform(Xtr); Wdv=vw.transform(Xdv); Wte=vw.transform(Xte)
# control 1: shuffled training labels
rng=np.random.default_rng(1); Ysh=Ytr[rng.permutation(len(Ytr))]
Sd,St=fitscore("svm",0.2,None,Wtr,Ysh,[Wdv,Wte]); m,_,_=evalS(Sd,Ydv,St,Yte); print("SHUFFLED labels ->",{k:round(v,4) for k,v in m.items()})
# control 2: real labels, independent metric recomputation (manual F1 from counts)
Sd,St=fitscore("svm",0.2,None,Wtr,Ytr,[Wdv,Wte]); m,Pg,Pc=evalS(Sd,Ydv,St,Yte); print("REAL ->",{k:round(v,4) for k,v in m.items()})
tp=(Pg&(Yte==1)).sum(0); fp=(Pg&(Yte==0)).sum(0); fn=((~Pg)&(Yte==1)).sum(0)
f=2*tp/np.maximum(2*tp+fp+fn,1); print("manual macro-F1:",round(float(f.mean()),4), "classes with F1=0:",int((f==0).sum()))
# control 3: overlap between splits
st=set(Xtr); print("exact text overlap train∩test:",sum(t in st for t in Xte),"train∩dev:",sum(t in st for t in Xdv))
# control 4: default threshold 0 for SVM, no tuning
P0=fix(St>=0,St); print("SVM default threshold macro-F1:",round(mf(Yte,P0,"macro"),4))
