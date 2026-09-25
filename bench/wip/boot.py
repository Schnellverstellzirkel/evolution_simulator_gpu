import sys, numpy as np
A=np.fromfile(sys.argv[1],dtype=np.float32).reshape(-1,4)[:,0].astype(np.float64)
for p in sys.argv[2:]:
    B=np.fromfile(p,dtype=np.float32).reshape(-1,4)[:,0].astype(np.float64)
    d=B-A
    # paired: same creatures, so test mean of per-creature differences
    se=d.std()/np.sqrt(len(d))
    print(f"{p.split('/')[-1]}: mean diff {d.mean():+.5f} (paired SE {se:.5f}, z {d.mean()/se:+.2f}), corr {np.corrcoef(A,B)[0,1]:.3f}, mean {A.mean():.4f} -> {B.mean():.4f}")
