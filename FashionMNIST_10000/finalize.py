from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from scipy.optimize import linear_sum_assignment
root=Path(__file__).resolve().parent
folder=root/'results'
runs=pd.read_csv(folder/'runs.csv');summary=pd.read_csv(folder/'summary.csv');pairs=pd.read_csv(folder/'pairwise_ari.csv')
with gzip.open(root/'data'/'t10k-labels-idx1-ubyte.gz','rb') as f: y=np.frombuffer(f.read(),dtype=np.uint8,offset=8)
assert len(y)==10000 and len(runs)==21 and runs.groupby('method').size().eq(3).all()
# Double-check every metric from stored predicted clusters.
for _,row in runs.iterrows():
 v=row['method'];s=int(row['seed']);suffix=v.replace(' ','_').replace('(','').replace(')','').replace('=','')
 fn=folder/(f'pred_{suffix}_seed{s}.npy' if v!='AE + KMeans (control)' else f'pred_AE_Kmeans_seed{s}.npy')
 assert fn.exists(),fn
 pred=np.load(fn)
 assert len(pred)==10000
 k=max(int(y.max()),int(pred.max()))+1
 w=np.zeros((k,k),dtype=np.int64); np.add.at(w,(pred,y),1)
 ri,ci=linear_sum_assignment(w.max()-w)
 acc=100*w[ri,ci].sum()/len(y)
 nmi=100*normalized_mutual_info_score(y,pred);ari=100*adjusted_rand_score(y,pred)
 assert abs(acc-row.ACC)<1e-3 and abs(nmi-row.NMI)<1e-3 and abs(ari-row.ARI)<1e-3,(fn,row,acc,nmi,ari)
print('PASS: recomputed 21 rows using 10k predictions each.')
# Figure: absolute values and sample standard deviations.
order=['DEC','IDEC','DECRA','IDEC + BRB-inspired','DCSS (sigma=1)','DCSS (RMS)']
d=summary.set_index('method').loc[order]
fig,axs=plt.subplots(1,3,figsize=(15,5.4))
for ax,metric in zip(axs,['acc','nmi','ari']):
 mean=d[f'{metric}_mean'];std=d[f'{metric}_std'];ax.barh(order,mean,xerr=std,capsize=3)
 ax.invert_yaxis();ax.set_xlim(0,63);ax.set_xlabel(f'{metric.upper()} (%)');ax.grid(axis='x',alpha=.2)
 for i,(m,s) in enumerate(zip(mean,std)):
  ax.text(m+1.0,i,f'{m:.1f}',va='center',fontsize=8)
axs[1].set_yticklabels([]);axs[2].set_yticklabels([])
fig.suptitle('Fashion-MNIST test 10,000 | shared AE | 3 seeds (mean +/- SD)')
fig.text(.5,.01,'Independent mechanism adaptations (not exact author-code reproductions); DCSS RMS is an ablation.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.04,1,.94]); fig.savefig(root/'Bieu_do_10k.png',dpi=170,bbox_inches='tight');plt.close(fig)
# Export report-ready table and pairwise means
stab=pairs.groupby('method').pairwise_ARI.agg(['mean','std']).rename(columns={'mean':'pairwise_ari_mean','std':'pairwise_ari_std'})
report=summary.set_index('method').join(stab)
cols=['n_seeds','acc_mean','acc_std','nmi_mean','nmi_std','ari_mean','ari_std','pairwise_ari_mean','pairwise_ari_std','train_time_mean_sec']
report[cols].round(4).to_csv(root/'Bang_so_lieu_de_chen_Word.csv')

mn=report.loc['DCSS (RMS)'];dcr=report.loc['DECRA'];idec=report.loc['IDEC'];dec=report.loc['DEC']
markdown=f'''# Kết quả thực nghiệm Fashion-MNIST 10.000 ảnh (trực tiếp, PyTorch CPU)

**Nguồn dữ liệu:** Fashion-MNIST, 10.000 mẫu test, 28×28 grayscale, 10 loại trang phục, dữ liệu gốc được đính kèm. Tải chính thức: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion .

**Phân biệt:** bản chuyển thể cơ chế trên AE chung, **KHÔNG** tái lập chính xác mọi mã tác giả. `IDEC+BRB-inspired` là bản minh họa reset/recluster, không phải BRB chính thức. `DCSS (RMS)` là ablation khởi tạo RBF theo dữ liệu, không phải mặc định chuẩn trên mọi benchmark.

35 epoch pretrain, 30 epoch fine-tune, 3 seed 0–2, Adam 0.001, batch 256, CPU; tất cả chạy trên toàn bộ 10.000 ảnh với nhãn chỉ dùng tính metrics hậu kiểm. Mean±sample SD (thang %). Thời gian chỉ fine-tune, không bao gồm pretrain.

| Cấu hình | ACC (%) | NMI (%) | ARI (%) | Pairwise ARI | Fine-tune (giây) |
|---|---:|---:|---:|---:|---:|
'''
for idx in ['DEC','IDEC','DECRA','IDEC + BRB-inspired','DCSS (sigma=1)','DCSS (RMS)']:
 r=report.loc[idx]
 markdown+=f'| {idx} | {r.acc_mean:.2f} ± {r.acc_std:.2f} | {r.nmi_mean:.2f} ± {r.nmi_std:.2f} | {r.ari_mean:.2f} ± {r.ari_std:.2f} | {r.pairwise_ari_mean:.4f} | {r.train_time_mean_sec:.2f} |\n'
markdown+=f'''
**Diễn giải:** DCSS (RMS) đứng đầu trung bình ở cả ACC ({mn.acc_mean:.2f}%), NMI ({mn.nmi_mean:.2f}%), ARI ({mn.ari_mean:.2f}%) và Pairwise ARI ({mn.pairwise_ari_mean:.4f}) trong **giao thức cụ thể này**. So với DECRA, chênh trung bình ACC +{mn.acc_mean-dcr.acc_mean:.2f} điểm %, NMI +{mn.nmi_mean-dcr.nmi_mean:.2f} điểm %, ARI +{mn.ari_mean-dcr.ari_mean:.2f} điểm %. So với IDEC, DCSS (RMS) cao hơn NMI +{mn.nmi_mean-idec.nmi_mean:.2f} và ARI +{mn.ari_mean-idec.ari_mean:.2f} điểm %; ACC cao hơn +{mn.acc_mean-idec.acc_mean:.2f} điểm %. Với sigma=1, DCSS giảm mạnh — bằng chứng độ nhạy khởi tạo, không phải bằng chứng chất lượng phương pháp gốc.

**Giới hạn:** chỉ 3 seed và tập Fashion-MNIST test không nhãn để học cụm (transductive); không có tập đánh giá chưa nhìn thấy, kết quả không chứng minh DCSS SOTA năm 2026 hoặc ưu thế phổ quát. Các số liệu BRB-inspired không được gán cho BRB của Miklautz và cộng sự. Thời gian AE tiền huấn luyện chung tính riêng trong runs.csv (khoảng 11–12s mỗi seed). Cần nhiều seed và xác minh bằng mã tác giả trước khi khẳng định vượt trội một cách đáng tin cậy.

**Chỉ số:** ACC (%) sau ghép nhãn Hungarian; NMI (%) mức chia sẻ thông tin; ARI (%) mức phù hợp cặp mẫu đã hiệu chỉnh ngẫu nhiên; Pairwise ARI (thang 0–1) đồng thuận phân hoạch giữa 3 seed. Độ lớn của SD với 3 seed chưa phản ánh đầy đủ phân phối biến động.

**Minh chứng:** `results/runs.csv` (21 lượt = 7 cấu hình × 3 seed), `results/summary.csv`, `results/pairwise_ari.csv`, file dự đoán `pred_*.npy`, mã thực thi `run_fashion_10k.py`, notebook và log `run.log`. Đã tính lại các metric từ pred và nhãn raw để xác nhận CSV nhất quán.
'''
(root/'Ket_qua_va_dien_giai.md').write_text(markdown,encoding='utf8')

zip_path=root.parent/'Ket_qua_10k_FashionMNIST_DEC_IDEC_DECRA_DCSS_BRB.zip'
with ZipFile(zip_path,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
 for p in root.rglob('*'):
  if not p.is_file():continue
  if 'weights_' in p.name or p.suffix=='.pt' or '__pycache__' in p.parts or p.name=='benchmark_cpu.py' or p.name=='run.log' and False:continue
  if p.name=='fashion_mnist_10k_raw_uint8.npz':continue # source raw files already in data/
  z.write(p,arcname=str(Path('FashionMNIST_10000')/p.relative_to(root)))
print('ZIP:',zip_path,'bytes',zip_path.stat().st_size)
print('summary:',(root/'Bang_so_lieu_de_chen_Word.csv').stat().st_size)
print('chart:',(root/'Bieu_do_10k.png').stat().st_size)
