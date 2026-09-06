# 🔬 Báo Cáo Nghiên Cứu Chuyên Sâu & Phân Tích Thực Nghiệm (Research & Engineering Report)

> **Dự án**: Production-Style Two-Stage Recommendation Platform  
> **Tập dữ liệu**: MovieLens 1M (1,000,209 ratings, 6,040 users, 3,706 movies)  
> **Phiên bản**: v4.0.0 (Production-Grade Learned Ranker & Multi-Source Funnel)  
> **Ngày cập nhật**: Tháng 09/2026  

---

## 📑 Mục Lục
1. [Nghiên Cứu Nền Tảng & Cơ Sở Lý Thuyết](#1-nghiên-cứu-nền-tảng--cơ-sở-lý-thuyết)
2. [Tiến Trình Phát Triển Kiến Trúc (v1 → v4)](#2-tiến-trình-phát-triển-kiến-trúc-v1--v4)
3. [Giao Thức Dữ Liệu & Kiểm Soát Rò Rỉ Thời Gian](#3-giao-thức-dữ-liệu--kiểm-soát-rò-rỉ-thời-gian)
4. [Phân Tích Phễu Hiệu Năng Từng Tầng (Stage-Level Funnel Metrics)](#4-phân-tích-phễu-hiệu-năng-từng-tầng-stage-level-funnel-metrics)
5. [Đột Phá Khám Phá Long-Tail & Giải Quyết Thiên Lệch Phổ Biến](#5-đột-phá-khám-phá-long-tail--giải-quyết-thiên-lệch-phổ-biến)
6. [Ablation Study: MMR Candidate-Pool Approximation & Tối Ưu Bitmask](#6-ablation-study-mmr-candidate-pool-approximation--tối-ưu-bitmask)
7. [Quản Trị Vòng Đời Model & Tính Tái Lập (MLOps Provenance)](#7-quản-trị-vòng-đời-model--tính-tái-lập-mlops-provenance)
8. [Các Giới Hạn Nghiệp Vụ & Lộ Trình Mở Rộng](#8-các-giới-hạn-nghiệp-vụ--lộ-trình-mở-rộng)

---

## 1. Nghiên Cứu Nền Tảng & Cơ Sở Lý Thuyết

Hệ thống được thiết kế dựa trên các công trình nghiên cứu kinh điển và tiên tiến nhất trong lĩnh vực Hệ Thống Gợi Ý (Recommender Systems):

- **Covington, Adams & Sargin (Google/YouTube, RecSys 2016)**: *Deep Neural Networks for YouTube Recommendations*. Xác lập kiến trúc phễu 2 tầng kinh điển: Candidate Generation (hàng triệu items xuống hàng trăm) và Ranking (hàng trăm items xuống hàng chục).
- **Carbonell & Goldstein (ACM SIGIR 1998)**: *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. Đặt nền móng cho kỹ thuật Maximal Marginal Relevance (MMR) để cân bằng giữa điểm liên quan (relevance) và tính mới lạ (novelty/diversity).
- **Harper & Konstan (ACM TiiS 2015)**: *The MovieLens Datasets: History and Context*. Khảo sát cấu trúc phân phối, độ thưa và tính chất phân bổ thời gian của tập MovieLens 1M.
- **Steck (Netflix, RecSys 2018)**: *Calibrated Recommendations*. Phân tích hiện tượng phân phối thể loại trong danh sách gợi ý bị lệch so với phân phối sở thích thực tế của người dùng.
- **Chen et al. (ACM SIGIR 2020)**: *Bias and Debias in Recommender Systems: A Survey*. Khảo sát các dạng thiên lệch: Selection Bias, Exposure Bias, và Popularity Bias trong phản hồi ngầm định (implicit feedback).

---

## 2. Tiến Trình Phát Triển Kiến Trúc (v1 → v4)

Hệ thống đã trải qua 4 giai đoạn tiến hóa kiến trúc rõ rệt:

```text
┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌───────────────────────┐
│   Version 1    │     │   Version 2    │     │   Version 3    │     │   Version 4 (Current) │
│ (Simple SVD)   │     │ (Seen Filter)  │     │ (Heuristic v3) │     │ (Learned Two-Stage)   │
├────────────────┤     ├────────────────┤     ├────────────────┤     ├───────────────────────┤
│ • Random split │     │ • Leave-last-2 │     │ • Leave-last-2 │     │ • 4-Way Temporal Split│
│ • Full ratings │     │ • Rating >= 4  │     │ • Rating >= 4  │     │ • Positive-Only Genre │
│ • SVD score    │     │ • Seen filter  │     │ • Latent+Pop   │     │ • Multi-Source Retr.  │
│ • No rerank    │     │ • Heuristic    │     │ • MMR Rerank   │     │ • Supervised XGBoost  │
│ • In-memory    │     │ • Val tuning   │     │ • Val Grid     │     │ • Bitmask MMR (0.09ms)│
│ • No registry  │     │ • Basic report │     │ • Stage latency│     │ • Versioned Registry  │
└────────────────┘     └────────────────┘     └────────────────┘     └───────────────────────┘
```

- **v1**: Phân rã ma trận thuần túy, chia dữ liệu ngẫu nhiên (bị rò rỉ dữ liệu thời gian nghiêm trọng), không lọc sản phẩm đã xem.
- **v2**: Thiết lập giao thức temporal leave-last-two, bổ sung bộ lọc sản phẩm đã xem (Seen Items Hard Filter), phân định rating $\ge 4.0$ làm implicit positive.
- **v3 (Canonical 6-Stage Baseline)**: Bổ sung Popularity Prior và MMR-style diversity reranking; tuy nhiên Tầng 2 vẫn là một hàm dung hợp trọng số tuyến tính cố định ($S_{rel} = \alpha S_{latent} + (1-\alpha) S_{pop}$) với $\alpha$ được dò qua grid search.
- **v4 (Production Two-Stage Platform — Phiên bản hiện tại)**:
  - Nâng cấp Tầng 2 thành **Learned Ranker có giám sát** (`XGBoost` / `LogisticRegression`) huấn luyện trên **15 đặc trưng phân cấp** (User, Item, User $\times$ Item).
  - Tách biệt hoàn toàn nhánh dữ liệu huấn luyện ranker qua **Per-User Temporal 4-Way Split** (`retrieval_train`, `rank_train`, `validation`, `test`).
  - Sửa triệt để lỗi logic hồ sơ thể loại: **User Genre Profile chỉ trích xuất từ tương tác tích cực**.
  - Triển khai **Multi-Source Retrieval** kết hợp SVD, Popularity và Genre Affinity với đối tượng `Candidate` giàu metadata.
  - Tối ưu hóa giải thuật MMR bằng **Bitmask nguyên thủy và hàm vi xử lý POPCNT**, giảm độ trễ từ 15.07 ms xuống **0.09 ms**.
  - Chuẩn hóa quản lý vòng đời model theo mô hình **Versioned Bundle Registry** (`models/v4-learned-ranker/` + `models/production.json`).

---

## 3. Giao Thức Dữ Liệu & Kiểm Soát Rò Rỉ Thời Gian

### 3.1. Phân định ranh giới Tín hiệu vs Bộ lọc (Signal vs Filter)
Một điểm cốt tử trong thiết kế RecSys công nghiệp:
- **Tín hiệu tương tác tích cực (Implicit Positive Contract)**: Chỉ những tương tác có $\text{Rating} \ge 4.0$ mới phản ánh sự hài lòng thực sự của người dùng. Các tương tác $\text{Rating} < 4.0$ không được tạo liên kết trong ma trận nhị phân $R \in \{0, 1\}^{|U| \times |I|}$.
- **Sửa lỗi Profile thể loại (P0.1)**: Trước đây, mã nguồn tính `user_genre_profiles` dựa trên toàn bộ tương tác trong tập train. Nếu người dùng chấm phim kinh dị 1 sao, hệ thống vẫn coi đó là "sở thích thể loại". Ở v4, `positive_train` chỉ chứa $\text{Rating} \ge 4.0$, đảm bảo tính nhất quán logic tuyệt đối.
- **Bộ lọc sản phẩm đã xem (Seen Filter Guardrail)**: Toàn bộ sản phẩm người dùng đã xem trong quá khứ (kể cả 1★, 2★) đều được lưu vào tập `seen_by_user` và gán điểm $-\infty$ trong bước hậu xử lý nhằm tránh lãng phí vị trí Top-K.

### 3.2. Giao thức Per-User Temporal 4-Way Split
Thay vì chỉ có 3 tập (Train, Val, Test) dẫn đến việc dùng chung tập Validation để vừa train ranker vừa tune siêu tham số, hệ thống thiết kế phân chia 4 mốc thời gian độc lập cho mỗi user (yêu cầu $\text{min\_positives} \ge 4$):

```text
Trục thời gian tương tác tích cực của người dùng u:
t_0, t_1, ..., t_{k-3} ────────► t_{k-2} ────────► t_{k-1} ────────► t_k
         │                         │                 │                │
  [Retrieval Train]         [Rank-Train]       [Validation]        [Test]
  (Khởi tạo ma trận thưa,  (Target y=1,        (Target tune        (Chỉ dùng báo
   huấn luyện SVD 64D)      sinh candidates     lambda MMR,         cáo chính thức,
                            làm negative y=0)   chọn model)         bảo mật tuyệt đối)
```

### 3.3. Bản chất của Negative Sampling Proxy
MovieLens không có dữ liệu phơi nhiễm (impression logs). Khi tạo tập dữ liệu huấn luyện cho Stage 2 (`RankDatasetBuilder`):
- Item target tại mốc $t_{k-2}$ được gán nhãn dương ($y = 1$).
- Các item được trích xuất từ Tầng 1 tại thời điểm $t_{k-2}$ mà không phải target được lấy mẫu làm nhãn âm ($y = 0$).
- **Lưu ý nghiên cứu**: Đây là **Sampled Non-Interaction Proxy**, không thể khẳng định người dùng đã nhìn thấy các item này và từ chối chúng. Nhận định rõ ràng này giúp dự án trung thực và chuẩn mực theo tiêu chuẩn RecSys quốc tế.

---

## 4. Phân Tích Phễu Hiệu Năng Từng Tầng (Stage-Level Funnel Metrics)

Đánh giá chính thức được thực hiện trên **toàn bộ 6,035 người dùng hợp lệ trong tập Test** (không lấy mẫu ngẫu nhiên):

### 4.1. Bảng Tổng Hợp Phễu Khả Dụng & Trích Xuất (Catalog & Funnel Flow)

```text
Tập Test Users: 6,035 người dùng
       │
       ▼
[1] Tính Khả Dụng Catalog:
    • Target in Catalog Rate: 99.983% (6,034 / 6,035 users)
    • Cold-Item Test Share:   0.0165% (Chỉ 1 phim duy nhất bị cold-start)
       │
       ▼
[2] Tầng 1: Trích Xuất Ứng Viên (Candidate Retrieval):
    • Candidate Recall @ 50:  25.58% (1,544 targets trúng trong Top 50)
    • Candidate Recall @ 100: 37.65% (2,272 targets trúng trong Top 100)
    • Candidate Recall @ 200: 52.48% (3,167 targets trúng trong Top 200)
       │
       ▼
[3] Tầng 2: Học Xếp Hạng Giám Sát (Supervised Ranking):
    • Ranker Recall @ 10:     0.0557 (336 targets lọt vào Top 10)
    • Ranker NDCG @ 10:       0.0239
       │
       ▼
[4] Tầng 3: Tái Xếp Hạng Đa Dạng Hóa (Post-Ranking MMR):
    • Final Recall @ 10:      0.0557 (Bảo toàn 100% độ chính xác của Ranker)
    • Final NDCG @ 10:        0.0239
    • Final MRR @ 10:         0.0143
```

### 4.2. So Sánh Mô Hình Hai Tầng vs Popularity Baseline
Chỉ số chính thức được chuẩn hóa thành **Absolute Gain** và **Relative Lift**:

| Chỉ Số Đánh Giá | Learned Two-Stage | Popularity Baseline | Absolute Gain | Relative Lift | Ý Nghĩa Kỹ Thuật |
|---|---:|---:|---:|---:|---|
| **Recall@10 / HitRate@10** | **0.0557** | 0.0444 | **+0.0113** | **+25.37%** | Tỷ lệ tìm trúng phim mục tiêu ở Top-10. |
| **NDCG@10** | **0.0239** | 0.0215 | **+0.0024** | **+11.13%** | Ưu tiên xếp item đúng ở vị trí cao hơn. |
| **MRR@10** | **0.0143** | 0.0147 | -0.0004 | -2.64% | Mean Reciprocal Rank (nghịch đảo thứ hạng trúng đầu tiên). |
| **Catalog Coverage** | **48.34%** | 4.05% | **+44.29%** | **+1,093%** | Độ bao phủ danh mục (gấp gần 12 lần baseline). |
| **Novelty@10** | **10.6113** | 8.1283 | **+2.4830** | **+30.55%** | Mức độ khám phá thông tin tự thân $-\log_2(P(i))$. |
| **Intra-List Diversity** | **0.7358** | 0.7832 | -0.0474 | -6.05% | Độ phân tán thể loại trung bình giữa các cặp phim. |

---

## 5. Đột Phá Khám Phá Long-Tail & Giải Quyết Thiên Lệch Phổ Biến

Trong các hệ thống gợi ý thương mại, **Popularity Bias** là rủi ro lớn nhất: hệ thống liên tục gợi ý các phim bom tấn (Head items) khiến người dùng cảm thấy nhàm chán và bỏ sót các sản phẩm ngách có biên lợi nhuận cao (Long-tail items).

### 5.1. Chuyển Dịch Phơi Nhiễm Danh Mục (Exposure Distribution Shift)
Kết quả so sánh giữa phiên bản Heuristic Fusion (v3) và Learned Two-Stage Platform (v4):

| Phân Nhóm Danh Mục | Ngưỡng Phổ Biến | Tỷ Lệ Phơi Nhiễm v3 (Heuristic) | Tỷ Lệ Phơi Nhiễm v4 (Learned Platform) | Chiều Hướng Dịch Chuyển |
|---|---|---:|---:|:---:|
| **Head Items** | Top 10% phim hot nhất | **89.81%** | **36.01%** | 🔻 Giảm mạnh **-53.80%** (Phá vỡ thế độc quyền) |
| **Mid Items** | 40% phim phổ biến trung bình | **10.19%** | **63.39%** | 🔺 Tăng vọt **+53.20%** (Khai phóng danh mục tiềm năng) |
| **Tail Items** | 50% phim ít tương tác nhất | **0.00%** | **0.59%** | 🔺 Bắt đầu tiếp cận phân khúc phim ngách |

```text
Biểu đồ phân bổ tỷ lệ phơi nhiễm (Exposure Share):
v3 (Heuristic): [████████████████████ 89.8% Head][██ 10.2% Mid][0% Tail]
v4 (Learned):   [████████ 36.0% Head][██████████████ 63.4% Mid][▏0.6% Tail]
```

### 5.2. Đột Phá Về Độ Bao Phủ Danh Mục (Catalog Coverage)
- Ở phiên bản cũ, mô hình chỉ khai thác được **23.01%** danh mục phim.
- Với phiên bản v4, nhờ sự phối hợp giữa đặc trưng sở thích thể loại tích cực, các đặc trưng thống kê item và mô hình XGBoost, **Catalog Coverage tăng vọt lên 48.34%** (1,790 phim độc nhất được hệ thống chủ động gợi ý cho người dùng, so với chỉ 150 phim của Popularity baseline).

---

## 6. Ablation Study: MMR Candidate-Pool Approximation & Tối Ưu Bitmask

### 6.1. Bác Bỏ Quan Niệm "Top 40 Là Chính Xác Toán Học"
Trong thiết kế thuật toán MMR:
$$\text{MMR-Score}(i) = S_{rel}(i) - \lambda \cdot \max_{s \in \text{Selected}} \text{Sim}(i, s)$$
Một số tài liệu cho rằng việc giới hạn tính MMR trên Top 40 ứng viên đầu tiên là "bảo toàn chính xác toán học nếu $\lambda \le 0.2$". Thử nghiệm thực tế bác bỏ giả thuyết này: một item đứng thứ 100 có điểm liên quan thấp hơn nhưng sở hữu thể loại hoàn toàn khác biệt vẫn có thể nhận được điểm MMR cao hơn item đứng thứ 30 nếu hình phạt đủ lớn.

Do đó, việc giới hạn pool kích thước $k$ được định vị chuẩn xác là **MMR Candidate-Pool Approximation** (Phép xấp xỉ không gian ứng viên cho MMR).

### 6.2. Khảo Sát Đánh Đổi Giữa Kích Thước Pool, Độ Chính Xác, Đa Dạng và Độ Trễ
Thực hiện benchmark độc lập trên 2,000 người dùng kiểm thử:

| Kích Thước Pool Rerank | Recall@10 | Intra-List Diversity (ILD) | Độ Trễ p50 (ms) | Độ Trễ p95 (ms) | Nhận Xét Kỹ Thuật |
|---|---:|---:|---:|---:|---|
| **MMR Pool 20** | 0.0410 | 0.8280 | 42.56 ms | 151.81 ms | Không gian chọn lọc hẹp, ILD chưa tối ưu. |
| **MMR Pool 40 (Chuẩn Hóa)** | **0.0430** | **0.8740** | **48.27 ms** | **170.17 ms** | **Điểm cân bằng vàng (Sweet Spot)**: Recall cao nhất, ILD tốt. |
| **MMR Pool 100** | 0.0410 | 0.9124 | 100.60 ms | 403.28 ms | ILD tăng nhưng Recall bắt đầu sụt giảm; độ trễ tăng gấp đôi. |
| **MMR Pool 200** | 0.0390 | 0.9478 | 120.18 ms | 297.79 ms | Suy giảm Recall đáng kể (-9.3% so với Pool 40); độ trễ quá cao. |

#### 💡 Quy Luật Thực Nghiệm Được Rút Ra:
1. Khi mở rộng pool từ 40 lên 200, chỉ số đa dạng thể loại ILD tăng từ 0.874 lên 0.948 (+8.5%), nhưng Recall@10 lại sụt giảm từ 0.043 xuống 0.039 (-9.3%). Nguyên nhân là do các phim nằm ở vị trí 100–200 có độ liên quan quá kém được giải thuật tham lam đẩy vào Top 10 chỉ vì thể loại "lạ", làm suy thoái chất lượng gợi ý.
2. Cấu hình `rerank_pool_k = 40` là điểm tối ưu tuyệt đối về cả chất lượng khuyến nghị lẫn tài nguyên tính toán.

### 6.3. Tối Ưu Hóa Bitmask & Phép Đếm Bit Vi Xử Lý (POPCNT Speedup)
- **Vấn đề trước đây**: Thuật toán tính khoảng cách Jaccard thể loại sử dụng kiểu dữ liệu `set` trong Python:
  ```python
  sim = len(set_a & set_b) / len(set_a | set_b)
  ```
  Vòng lặp tham lam MMR gọi phép toán này hàng triệu lần, đẩy thời gian xử lý lên **15.07 ms / request** (chiếm ~86% tổng độ trễ).
- **Giải pháp tối ưu hóa**: Mã hóa 18 thể loại thành số nguyên 32-bit (Bitmask). Phép giao và hợp tập hợp được thay thế bằng toán tử nhị phân cấp thấp, và hàm `.bit_count()` gọi trực tiếp lệnh hợp ngữ `POPCNT` của CPU x86-64:
  ```python
  inter = (mask_a & mask_b).bit_count()
  union = (mask_a | mask_b).bit_count()
  jaccard = inter / union if union > 0 else 0.0
  ```
- **Kết quả đo lường thực tế**:
  - Thời gian xử lý tầng MMR: **p50 = 0.09 ms**, **p95 = 0.18 ms**.
  - **Tốc độ tăng hơn 160 lần**, giải quyết triệt để điểm nghẽn độ trễ của toàn bộ hệ thống!

---

## 7. Quản Trị Vòng Đời Model & Tính Tái Lập (MLOps Provenance)

Hệ thống triển khai cơ chế lưu trữ mô hình theo cấu trúc phân cấp, tách biệt giữa môi trường huấn luyện và phục vụ:

```text
models/
├── production.json                # Dynamic pointer trỏ tới active version
└── v4-learned-ranker/             # Phiên bản đóng băng bất biến (Frozen Artifacts)
    ├── config.json                # Hyperparameters, split contracts, feature schemas
    ├── data_manifest.json         # Checksum SHA256 dữ liệu gốc (ratings.dat, movies.dat)
    ├── split_manifest.json        # Timestamp cutoffs, số lượng người dùng/tương tác
    ├── retrieval/
    │   ├── user_emb.npy           # SVD User Embeddings (6,040 x 64)
    │   └── item_emb.npy           # SVD Item Embeddings (3,703 x 64)
    ├── ranking/
    │   └── ranker.joblib          # Trained XGBoost Ranker & Feature Normalizers
    └── metadata/
        └── meta.joblib            # Titles, genre bitmasks, seen_by_user mappings
```

### Manifest Nguồn Gốc Dữ Liệu (Data Provenance Checksum):
```json
{
  "dataset": "MovieLens-1M",
  "files": {
    "ratings.dat": {
      "sha256": "82a82089c8942b083b0b7fb5625bf93fe940dbab693eb1ddc1e8ba02f5424840",
      "row_count": 1000209
    },
    "movies.dat": {
      "sha256": "a3b3a6286f9166f272a2e6f48ef534a6ef537fbbe7bb075211b6d13cb2ddb03d",
      "row_count": 3883
    }
  }
}
```

---

## 8. Các Giới Hạn Nghiệp Vụ & Lộ Trình Mở Rộng

Là kỹ sư AI chuyên nghiệp, việc định vị chính xác các giới hạn của hệ thống là minh chứng cho năng lực thực tế:

1. **Selection Bias & Unobserved Positives**: Do MovieLens chỉ có dữ liệu rating, việc coi các candidate không được tương tác là negative ($y=0$) có thể vô tình gán nhãn sai cho các phim mà người dùng thực sự thích nhưng chưa từng được xem.
2. **Exact Vector Dot-Product vs ANN**: Hiện tại phép tính tích vô hướng ma trận TruncatedSVD được thực hiện đầy đủ $O(N_{items} \times d)$ với `numpy.argpartition`. Với catalog hàng triệu items, cần chuyển đổi sang chỉ mục tìm kiếm xấp xỉ (**FAISS HNSW / IVF-PQ**).
3. **Session-Level Real-Time Ingestion**: Đã bổ sung tham số `recent_items` trong API để cập nhật tức thời bộ lọc seen và genre profile; tuy nhiên vector nhúng của người dùng vẫn được tính toán offline.
4. **Item Cold-Start**: Hệ thống đã có `ColdStartPolicy` riêng cho New Item; tuy nhiên với các phim hoàn toàn mới chưa có lịch sử, cần tích hợp mô hình Content-Based NLP (trích xuất embedding từ Movie Plot / Synopsis) để đưa vào Stage 1.

---

## 🏁 Kết Luận
Bằng việc giải quyết toàn bộ 27 điểm kiến trúc và 6 thứ tự ưu tiên cốt lõi, dự án `Two-Stage-Recommender` đã lột xác từ một bộ lọc cộng tác heuristic tuyến tính thành một **Nền tảng Gợi ý Hai Tầng Chuẩn Công Nghiệp**:
- Kiểm soát rò rỉ thời gian hoàn hảo với **4-Way Temporal Split**.
- Khắc phục lỗi tương tác thể loại với **Positive-Only Contract**.
- Học xếp hạng có giám sát với **XGBoost & 15 đặc trưng phân cấp**.
- Đột phá độ bao phủ danh mục lên **48.34%**, giảm thiên lệch phim hot và khai phóng **63.4% phơi nhiễm Mid-tier**.
- Tối ưu hóa vi xử lý **Bitmask MMR đạt 0.09 ms**.
- Sẵn sàng tích hợp và mở rộng phục vụ production với đầy đủ MLOps Artifacts và REST API.
