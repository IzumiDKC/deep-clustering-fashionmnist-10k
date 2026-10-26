# Fashion-MNIST 10.000 ảnh — thực nghiệm Data Mining

## Tính chất của kết quả

Đây là **thực nghiệm có chạy trực tiếp**, dùng **các bản chuyển thể độc lập trên một PyTorch autoencoder chung**. **KHÔNG phải** số liệu chạy nguyên mã tác giả DEC/IDEC/DECRA/BRB/DCSS và **KHÔNG** dùng để tuyên bố DCSS là tốt nhất trong mọi hoàn cảnh. Phương pháp `IDEC + BRB-inspired` chỉ áp dụng reset + reclustering mô phỏng ý tưởng BRB, không phải toàn bộ BRB nguyên gốc. `DCSS (RMS)` là một sửa đổi khởi tạo độ rộng RBF để thích ứng scale (ablation); `DCSS (sigma=1)` cũng được báo cáo công khai.

## Dữ liệu

- `data/t10k-images-idx3-ubyte.gz`: 10.000 ảnh 28×28 từ Fashion-MNIST **test split**, từng pixel trong [0,255].
- `data/t10k-labels-idx1-ubyte.gz`: nhãn 10 loại quần áo, chỉ dùng **sau** khi học cụm để tính ACC/NMI/ARI.
- Cùng toàn bộ 10.000 ảnh để học cụm **transductive**. Không có tập train/test có giám sát. Số cụm K=10 được xác định theo thiết kế benchmark.
- Nguồn chính thức: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
- Giá trị SHA-256 của hai file được ghi trong `results/dataset_manifest.json`.

## Cách tái chạy

### Windows / VS Code

Mở terminal tại thư mục gốc repository (thư mục cha của `FashionMNIST_10000`). Tải dữ liệu theo [hướng dẫn ở thư mục gốc](../README.md#chuẩn-bị-dữ-liệu) trước khi chạy. Dùng môi trường Python riêng để các lệnh chạy và notebook dùng cùng thư viện:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\FashionMNIST_10000\run_fashion_10k.py --seeds 3 --pretrain 35 --epochs 30
```

Ba lệnh đầu chỉ cần thực hiện khi tạo môi trường mới. Các lần chạy sau dùng lệnh cuối. Gọi trực tiếp `.venv\Scripts\python.exe` không cần kích hoạt môi trường và không phụ thuộc shortcut `python` của Windows. PyTorch CPU 2.10.0 khớp phiên bản ghi trong cấu hình kết quả cũ; các phiên bản thư viện khác và máy chạy vẫn có thể làm kết quả thay đổi.

Trong VS Code, chọn **Python: Select Interpreter** → `.venv\Scripts\python.exe`. Với notebook, chọn **Select Kernel** → **Python Environments** → môi trường `.venv` này. Kiểm tra môi trường bằng cách chạy `import sys; print(sys.executable)` trong notebook.

### Môi trường Python khác

Cài Python >=3.10 và thư viện: `pip install torch numpy pandas scipy scikit-learn matplotlib nbformat`. Tại thư mục này:

```bash
python run_fashion_10k.py --seeds 3 --pretrain 35 --epochs 30
```

Với CPU, thời gian chạy có thể từ vài phút đến nhiều phút tùy CPU. Thiết kế ở đây là encoder `784-256-128-10`, decoder đối xứng, Adam lr=0.001, batch=256, 35 epoch tiền huấn luyện trên từng seed, 30 epoch fine-tuning mỗi cấu hình. Chạy seed 0/1/2. Mỗi seed, toàn bộ 6 phương pháp dùng chung checkpoint AE và tâm K-means ban đầu. `AE+KMeans` là mốc không fine-tune.

Notebook `00_chay_va_xem_ket_qua.ipynb` cũng cung cấp nút để tái chạy và hiển thị kết quả. **Lưu ý**: nếu chạy lại, kết quả cũ trong `results/` bị ghi đè; sao lưu trước nếu cần đối chiếu.

## Diễn giải CSV

- `results/runs.csv`: từng thuật toán và seed, ACC/NMI/ARI (%) và thời gian huấn luyện fine-tune (không gồm pretraining).
- `results/summary.csv`: mean ± sample SD theo 3 seed, chưa đủ mạnh để làm kết luận thống kê toàn diện.
- `results/pairwise_ari.csv`: độ đồng thuận giữa các phân hoạch từ 3 seed, không cần nhãn.
- `results/trace_*.csv`: độ thay đổi kết quả qua các epoch.
- `results/pred_*.npy`: nhãn cụm dự đoán, có thể tính lại metrics từ raw labels.
- `results/config.json`: cấu hình thực nghiệm.

### Ý nghĩa chỉ số

ACC = % phân cụm đúng sau khi ghép nhãn cụm với lớp thật theo Hungarian. NMI/ARI = mức phù hợp giữa phân cụm dự đoán và nhãn tham chiếu, với mức cao thường tốt hơn. Pairwise ARI = mức phân hoạch đồng nhất giữa hai seed, không dựa vào nhãn thật. Thời gian `train_seconds` chỉ tính quá trình fine-tune, thời gian AE tiền huấn luyện (`pretrain_seconds`) là chi phí chung theo seed. So sánh đúng các phương pháp trong cùng bảng chạy, không gộp với con số trên MNIST hay nguồn khác.

## Ghi chú quan trọng về phương pháp

- DEC: fine-tune bằng KL mục tiêu tự làm sắc.
- IDEC: thêm reconstruction loss để giữ đặc trưng.
- DECRA-style: lớp biến đổi latent với reconstruction consistency, không đầy đủ cơ chế residual pretraining của tác giả. Bản hiện tại khởi tạo phép biến đổi đồng nhất nên consistency bằng 0 và nhánh biến đổi không học; cần sửa và chạy lại trước khi đánh giá tác dụng của nhánh này.
- DCSS-style: tối ưu reconstruction + soft silhouette + entropy, gồm sigma mặc định và sigma theo RMS.
- IDEC + BRB-inspired: thực nghiệm reset một phần trọng số encoder và chạy lại K-means định kỳ để minh họa cơ chế vượt bão hòa; **không gán kết quả này cho BRB bản gốc**.

Đừng làm biểu đồ kết quả tăng đều theo năm xuất bản; chất lượng tùy benchmark, khởi tạo và protocol.
