import sys, numpy as np
def load(p):
    a=np.fromfile(p,dtype=np.float32).reshape(-1,4); return a
names=['fitness','contact','vert_osc','gait_hz']
A=load(sys.argv[1]); B=load(sys.argv[2])
for i,n in enumerate(names):
    a=A[:,i]; b=B[:,i]
    fa=a[a>-1e19]; fb=b[b>-1e19]
    q=[1,10,50,90,99,99.9]
    print(f"{n:9s} failed {np.sum(a<=-1e19):5d}/{np.sum(b<=-1e19):5d} mean {fa.mean():.5f}/{fb.mean():.5f} " + " ".join(f"p{x}={np.percentile(fa,x):.4f}/{np.percentile(fb,x):.4f}" for x in q))
fa=A[:,0]; fb=B[:,0]
ok=(fa>-1e19)&(fb>-1e19)
print("fitness correlation", np.corrcoef(fa[ok],fb[ok])[0,1])
from scipy import stats
print("KS fitness", stats.ks_2samp(fa[ok],fb[ok]))
