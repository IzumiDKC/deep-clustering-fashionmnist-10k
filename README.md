# Deep Clustering — Fashion-MNIST 10.000 ảnh

Mã thực nghiệm PyTorch CPU cho 7 cấu hình: AE + KMeans, DEC, IDEC, DECRA-style, IDEC + BRB-inspired, DCSS khởi tạo sigma = 1 và DCSS khởi tạo RMS. Cấu hình mặc định dùng 3 seed, 35 epoch tiền huấn luyện và 30 epoch phân cụm.

Đây là các bản chuyển thể cơ chế dùng chung autoencoder, không phải bản tái lập đầy đủ mã tác giả. Chỉ so sánh kết quả trong cùng giao thức; không dùng để khẳng định thứ hạng phương pháp gốc.

## Nội dung repository

- `FashionMNIST_10000/run_fashion_10k.py`: chạy thực nghiệm trên terminal.
- `FashionMNIST_10000/00_chay_va_xem_ket_qua.ipynb`: xem bảng và biểu đồ sau khi chạy; có ô tùy chọn để chạy lại.
- `FashionMNIST_10000/finalize.py`: kiểm tra chỉ số từ các dự đoán của 21 lượt chạy, xuất bảng CSV, biểu đồ và gói kết quả cục bộ.
- `requirements.txt`: phiên bản các thư viện trực tiếp của môi trường demo.

Repository chỉ chứa mã và hướng dẫn. Dữ liệu, kết quả, trọng số, đầu ra notebook, môi trường Python, Word và PowerPoint không được đưa lên GitHub.

## Cài đặt trên Windows

Mở PowerShell tại thư mục gốc repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lệnh dùng trực tiếp Python trong `.venv`, không cần kích hoạt môi trường. Các phiên bản thư viện phụ thuộc gián tiếp và phần cứng vẫn có thể làm số liệu thay đổi giữa các máy.

## Chuẩn bị dữ liệu

Tải hai file test của [Fashion-MNIST chính thức](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) vào `FashionMNIST_10000/data/`, giữ nguyên định dạng `.gz`:

```powershell
New-Item -ItemType Directory -Force .\FashionMNIST_10000\data | Out-Null
curl.exe --fail --location "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz" --output .\FashionMNIST_10000\data\t10k-images-idx3-ubyte.gz
curl.exe --fail --location "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz" --output .\FashionMNIST_10000\data\t10k-labels-idx1-ubyte.gz
```

Dữ liệu gồm 10.000 ảnh xám 28 × 28, 10 lớp. Toàn bộ ảnh được dùng để học cụm theo chế độ transductive; nhãn chỉ dùng để tính chỉ số đánh giá, không tham gia loss.

## Chạy trên terminal

```powershell
.\.venv\Scripts\python.exe .\FashionMNIST_10000\run_fashion_10k.py --seeds 3 --pretrain 35 --epochs 30
```

Chương trình in tiến trình và bảng tổng hợp, đồng thời lưu đầu ra vào `FashionMNIST_10000/results/`. Lần chạy mới ghi đè các file kết quả cùng tên; sao lưu trước nếu muốn giữ lần chạy cũ.

Sau khi chạy đủ 3 seed, có thể xuất bảng và biểu đồ:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\FashionMNIST_10000\finalize.py
```

Các file xuất và gói ZIP chỉ được tạo trên máy, đã được loại khỏi Git bằng `.gitignore`.

## Xem notebook

Mở `FashionMNIST_10000/00_chay_va_xem_ket_qua.ipynb`, chọn **Select Kernel → Python Environments → .venv**. Chạy notebook với thư mục làm việc là `FashionMNIST_10000` để đọc đúng `data/` và `results/`. Ô gọi huấn luyện đang được comment; chạy terminal trước rồi dùng các ô còn lại để xem kết quả.

## Giới hạn của mã hiện tại

- Chỉ một bộ dữ liệu và ba seed; kết quả không đo khả năng tổng quát hóa trên ảnh chưa gặp.
- `IDEC + BRB-inspired` là minh họa reset/reclustering, không phải BRB đầy đủ.
- DECRA-style khởi tạo phép biến đổi latent bằng ma trận đơn vị và bias bằng 0. Trong cấu hình hiện tại, hai đầu ra tái tạo trùng nhau nên nhánh consistency không có tín hiệu học. Cần xử lý và chạy lại trước khi kết luận về tác dụng của cơ chế này.
- `DCSS (sigma=1)` chỉ nói về giá trị khởi tạo; sigma vẫn được tối ưu trong quá trình học.
- Thời gian `train_seconds` không bao gồm AE pretraining và KMeans khởi tạo; vòng đo có cả các lần đánh giá trong quá trình fine-tune. Giá trị 0 của đối chứng AE + KMeans không phải tổng thời gian chạy.

Xem thêm [thiết kế và cách đọc chỉ số](FashionMNIST_10000/README.md).
