"""Reproducible CPU benchmark on uploaded Fashion-MNIST test images, 10k transductive.
Independent common-backbone PyTorch mechanism adaptations; NOT author-code reproductions.
All outputs derived from direct computation and same images; ground truth labels used only for posthoc metrics.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,random,time,os,sys,copy
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
IMG=Path(os.environ.get('FASHION_IMAGES',str(ROOT/'data'/'t10k-images-idx3-ubyte.gz')))
LABEL=Path(os.environ.get('FASHION_LABELS',str(ROOT/'data'/'t10k-labels-idx1-ubyte.gz')))
OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)

def load():
    with gzip.open(IMG,'rb') as f: ib=f.read()
    with gzip.open(LABEL,'rb') as f: lb=f.read()
    magic,n,h,w=np.frombuffer(ib[:16],dtype='>u4')
    magic_y,n_y=np.frombuffer(lb[:8],dtype='>u4')
    assert (magic,n,h,w)==(2051,10000,28,28) and (magic_y,n_y)==(2049,10000)
    x=np.frombuffer(ib,offset=16,dtype=np.uint8).copy().reshape(n,h*w).astype(np.float32)/255.
    y=np.frombuffer(lb,offset=8,dtype=np.uint8).copy().astype(np.int64)
    assert len(x)==len(y)==10000 and len(np.unique(y))==10
    manifest={'dataset':'Fashion-MNIST official 10k test split','n_samples':len(x),'features':784,'classes':10,
        'image_sha256':hashlib.sha256(IMG.read_bytes()).hexdigest(), 'label_sha256':hashlib.sha256(LABEL.read_bytes()).hexdigest(),
        'labels_use':'posthoc evaluation only','file_names':[IMG.name,LABEL.name]}
    (OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    np.savez_compressed(OUT/'fashion_mnist_10k_raw_uint8.npz', images=(x*255).astype('uint8').reshape(-1,28,28),labels=y.astype('uint8'))
    return torch.from_numpy(x),y

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
        self.decoder=nn.Sequential(nn.Linear(10,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,784),nn.Sigmoid())
    def encode(self,x):return self.encoder(x)
    def forward(self,x):return self.decoder(self.encode(x))

class Clustering(nn.Module):
    def __init__(self, base:AE,centers:np.ndarray,variant:str,widths=None):
        super().__init__()
        self.ae=base
        self.centers=nn.Parameter(torch.from_numpy(centers.astype('float32')).clone())
        self.variant=variant
        if variant=='DECRA':
            self.zhat=nn.Linear(10,10)
            with torch.no_grad(): nn.init.eye_(self.zhat.weight); self.zhat.bias.zero_()
        if variant.startswith('DCSS'):
            if widths is None:widths=np.ones(10)
            self.log_sigmas=nn.Parameter(torch.from_numpy(np.log(np.clip(widths,1e-3,None)).astype('float32')))
    def q(self,z):
        dist2=(z[:,None,:]-self.centers[None,:,:]).pow(2).sum(-1)
        if self.variant.startswith('DCSS'):
            rbf=torch.exp(-dist2/torch.exp(2*self.log_sigmas)[None,:].clamp_min(1e-5))
            return torch.softmax(rbf/0.25,dim=1)
        num=(1+dist2).reciprocal()
        return num/num.sum(-1,keepdim=True)

def encode_all(ae,x,batch=2048):
    ae.eval()
    with torch.no_grad():return torch.cat([ae.encode(xx) for xx in x.split(batch)]).numpy()

def predict(model,x):
    model.eval()
    with torch.no_grad():return torch.cat([model.q(model.ae.encode(xx)).argmax(-1) for xx in x.split(2048)]).numpy()

def target_distribution(q):
    weight=(q*q)/(q.sum(0,keepdim=True)+1e-12)
    return weight/(weight.sum(1,keepdim=True)+1e-12)

def true_metrics(y,pr):
    K=max(int(y.max()),int(pr.max()))+1
    w=np.zeros((K,K),dtype=np.int64)
    np.add.at(w,(pr,y),1)
    r,c=linear_sum_assignment(w.max()-w)
    return {'ACC':round(100*w[r,c].sum()/len(y),5),'NMI':round(100*normalized_mutual_info_score(y,pr),5),
      'ARI':round(100*adjusted_rand_score(y,pr),5), 'occupied_clusters':len(set(pr)),
      'smallest_cluster':int(np.bincount(pr,minlength=10).min()), 'largest_cluster':int(np.bincount(pr,minlength=10).max())}

def soft_silhouette(z,q):
    """Follows author-code structure: alpha=cdist(z,z)@q; beta=min alpha over other clusters."""
    d=torch.cdist(z,z,p=2)
    alpha=d@q # matches author's source (not normalized by cluster mass)
    K=alpha.shape[1]
    alt=alpha[:,None,:].expand(-1,K,-1).clone()
    alt[:,torch.arange(K),torch.arange(K)]=float('inf')
    beta=alt.min(dim=-1).values
    si=(beta-alpha)/torch.maximum(alpha,beta).clamp_min(1e-8)
    return (q*si).sum(-1).mean()

def reset_and_recluster(model,x,seed,epoch):
    # BRB-inspired approximation: partial (10%) encoder weight reset + KMeans reclustering.
    state=AE().encoder
    torch.manual_seed(seed*1000+epoch+14093)
    for p in state.parameters():
        if p.ndim>1: nn.init.kaiming_uniform_(p,a=5**0.5)
        else: nn.init.zeros_(p)
    with torch.no_grad():
        for p,new_p in zip(model.ae.encoder.parameters(),state.parameters()):
            p.mul_(0.9).add_(new_p,alpha=0.1)
    embeddings=encode_all(model.ae,x)
    centers=KMeans(n_clusters=10,n_init=10,random_state=seed+epoch).fit(embeddings).cluster_centers_.astype('float32')
    with torch.no_grad():model.centers.copy_(torch.from_numpy(centers))

def one_variant(v,base,centers,cluster_labels,x,y,seed,epochs,batch=256):
    base=copy.deepcopy(base)
    if v=='DCSS (RMS)':
        emb=encode_all(base,x)
        d2=((emb-centers[cluster_labels])**2).sum(-1)
        widths=np.array([np.sqrt(max(d2[cluster_labels==k].mean(),1e-4)) if np.any(cluster_labels==k) else 1. for k in range(10)],dtype='float32')
    else: widths=None
    model=Clustering(base,centers,v,widths)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    gen=torch.Generator().manual_seed(seed*200+17)
    loader=DataLoader(TensorDataset(x,torch.arange(len(x))),batch_size=batch,shuffle=True,generator=gen)
    print(f'  start {v}',flush=True)
    start=time.perf_counter()
    trace=[]
    for epoch in range(epochs):
        if v=='IDEC + BRB-inspired' and epoch>0 and epoch%10==0:
            reset_and_recluster(model,x,seed,epoch)
        if v not in ['DCSS (RMS)','DCSS (sigma=1)'] and epoch%4==0:
            model.eval()
            with torch.no_grad():
                all_q=torch.cat([model.q(model.ae.encode(xx)) for xx in x.split(2048)],dim=0)
                p_all=target_distribution(all_q).detach()
        model.train()
        running=0
        for xb,indices in loader:
            opt.zero_grad(set_to_none=True)
            z=model.ae.encode(xb)
            q=model.q(z)
            if v.startswith('DCSS'):
                rec=F.mse_loss(model.ae.decoder(z),xb)
                silhouette=soft_silhouette(z,q)
                entr=-(q*(q+1e-12).log2()).sum(1).mean()
                loss=rec+0.01*(1-silhouette)+0.01*entr
            else:
                kl=F.kl_div((q+1e-12).log(),p_all[indices],reduction='batchmean')
                if v=='DEC':loss=kl
                elif v in ['IDEC','IDEC + BRB-inspired']:
                    rec=F.mse_loss(model.ae.decoder(z),xb)
                    loss=rec+0.1*kl
                elif v=='DECRA':
                    original_recon=model.ae.decoder(z)
                    modified_recon=model.ae.decoder(model.zhat(z))
                    rec=F.mse_loss(original_recon,xb)
                    consistency=F.mse_loss(modified_recon,original_recon)
                    loss=2*rec+0.6*consistency+0.1*kl
                else: raise ValueError(v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            opt.step()
            running+=float(loss.detach())
        if epoch in (0,9,19,epochs-1):
            pred=predict(model,x)
            trace.append({'seed':seed,'method':v,'epoch':epoch+1,'loss_sum':running,**true_metrics(y,pred)})
            print(f'    ep={epoch+1:3} loss={running:.3f} ACC={trace[-1]["ACC"]:.2f} NMI={trace[-1]["NMI"]:.2f} ARI={trace[-1]["ARI"]:.2f}',flush=True)
    secs=time.perf_counter()-start
    pred=predict(model,x)
    np.save(OUT/f'pred_{v.replace(" ","_").replace("(","").replace(")","").replace("=","")}_seed{seed}.npy',pred.astype('int16'))
    row={'method':v,'seed':seed,**true_metrics(y,pred),'train_seconds':round(secs,3), 'pretrain_seconds':None,
       'epochs':epochs,'status':'independent PyTorch mechanism adaptation, NOT official author reproduction'}
    pd.DataFrame(trace).to_csv(OUT/f'trace_{v.replace(" ","_").replace("(","").replace(")","").replace("=","")}_seed{seed}.csv',index=False)
    torch.save({'method':v,'seed':seed,'model_state_dict':model.state_dict()},OUT/f'weights_{v.replace(" ","_").replace("(","").replace(")","").replace("=","")}_seed{seed}.pt')
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--pretrain',type=int,default=35);ap.add_argument('--epochs',type=int,default=30); args=ap.parse_args()
    x,y=load();print('loaded',x.shape, 'label histogram',np.bincount(y),flush=True)
    config={'image_source':str(IMG),'label_source':str(LABEL),'dataset':'Fashion-MNIST, official test set 10k','pretrain_epochs':args.pretrain,
        'clustering_epochs':args.epochs,'batch':256,'learning_rate':1e-3,'seeds':list(range(args.seeds)),
        'encoder':'784-256-128-10','decoder':'10-128-256-784','losses':'DEC KL; IDEC recon+0.1KL; DECRA 2 recon + 0.6 consistency + 0.1 KL; DCSS recon + 0.01(1-soft silhouette) + 0.01 entropy',
        'dcss_sigma':'1 and RMS per cluster separately','BRB':'reset-and-recluster inspired (not official BRB implementation)',
        'evaluation':'transductive whole unlabeled test split; labels used only after training, K=10 known from dataset',
        'caveat':'Independent mechanism adaptations using common AE backbone, not exact author implementations',
        'torch':torch.__version__,'cpu_threads':torch.get_num_threads()}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    runs=[]
    p=OUT/'runs.csv'
    if p.exists():p.unlink()
    methods=['DEC','IDEC','DECRA','DCSS (sigma=1)','DCSS (RMS)','IDEC + BRB-inspired']
    for seed in range(args.seeds):
        print(f'\n### SEED={seed} ###',flush=True)
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
        pretrain=AE();optim=torch.optim.Adam(pretrain.parameters(),lr=0.001)
        loader=DataLoader(TensorDataset(x),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
        start=time.perf_counter()
        for ep in range(args.pretrain):
            pretrain.train();agg=0.
            for (xb,) in loader:
                optim.zero_grad(set_to_none=True)
                loss=F.mse_loss(pretrain(xb),xb)
                loss.backward();optim.step();agg+=float(loss.detach())
            if (ep+1)%10==0 or ep==0:print(' pretrain',ep+1,'MSE_loss_sum',round(agg,3),flush=True)
        pre_secs=time.perf_counter()-start
        torch.save(pretrain.state_dict(),OUT/f'pretrain_seed{seed}.pt')
        z=encode_all(pretrain,x)
        km=KMeans(n_clusters=10,n_init=20,random_state=seed).fit(z)
        pre_metrics=true_metrics(y,km.labels_)
        print(' shared AE+KMeans',pre_metrics,'pretrain seconds',round(pre_secs,2),flush=True)
        np.save(OUT/f'pred_AE_Kmeans_seed{seed}.npy',km.labels_.astype('int16'))
        base_row={'method':'AE + KMeans (control)','seed':seed,**pre_metrics,'train_seconds':0.,'pretrain_seconds':round(pre_secs,3),'epochs':0,
            'status':'common pretrained initial embedding baseline'}
        runs.append(base_row);pd.DataFrame(runs).to_csv(p,index=False)
        for variant in methods:
            row=one_variant(variant,pretrain,km.cluster_centers_,km.labels_,x,y,seed,args.epochs)
            row['pretrain_seconds']=round(pre_secs,3)
            runs.append(row);pd.DataFrame(runs).to_csv(p,index=False)
            print('  DONE',variant,'secs',row['train_seconds'],flush=True)
    df=pd.DataFrame(runs)
    stats=df.groupby('method').agg(n_seeds=('seed','count'),acc_mean=('ACC','mean'),acc_std=('ACC','std'),nmi_mean=('NMI','mean'),nmi_std=('NMI','std'),ari_mean=('ARI','mean'),ari_std=('ARI','std'),train_time_mean_sec=('train_seconds','mean')).reset_index()
    stats.to_csv(OUT/'summary.csv',index=False)
    print('\nSUMMARY\n',stats.to_string(index=False),flush=True)
    pairs=[]
    for method in df.method.unique():
        preds=[]
        fn=lambda seed,v=method: (OUT/f'pred_{v.replace(" ","_").replace("(","").replace(")","").replace("=","")}_seed{seed}.npy') if v!='AE + KMeans (control)' else OUT/f'pred_AE_Kmeans_seed{seed}.npy'
        for s in range(args.seeds):
            q=fn(s)
            if q.exists(): preds.append((s,np.load(q)))
        for (a,ya),(b,yb) in combinations(preds,2):
            pairs.append({'method':method,'seed_a':a,'seed_b':b,'pairwise_ARI':adjusted_rand_score(ya,yb)})
    pd.DataFrame(pairs).to_csv(OUT/'pairwise_ari.csv',index=False)
    print('All results saved at',OUT,flush=True)

if __name__=='__main__':main()
