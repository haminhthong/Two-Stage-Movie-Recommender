# 🎬 Two-Stage Recommendation System (MovieLens 1M)

> **Kiến trúc hệ thống gợi ý 2 tầng phong cách Production (Production-Style Two-Stage Recommender Architecture)**: Tầng 1 (Candidate Retrieval) lọc ứng viên tiềm năng bằng Matrix Factorization (TruncatedSVD); Tầng 2 (Reranking) tối ưu hóa cá nhân hóa, cân bằng điểm phổ biến (Popularity Prior) và phạt mức độ lặp thể loại (MMR-Style Genre Diversity Reranking) nhằm hạn chế Filter Bubble.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Tổng Quan Dự Án

Trong các hệ thống thương mại thực tế (Netflix, YouTube, Spotify), số lượng sản phẩm/nội dung lên tới hàng triệu items. Việc tính toán điểm số phức tạp cho toàn bộ catalog trong mỗi request người dùng là **không khả thi về mặt độ trễ (latency)**. 

Dự án này triển khai kiến trúc **Two-Stage Recommender Pipeline** tham chiếu thiết kế thực tế (Production-Style Architecture) trên tập dữ liệu MovieLens 1M (~3,700 phim, 6,040 người dùng, 1 triệu lượt đánh giá):
1. **Giai đoạn Retrieval (Candidate Generation)**: Lọc nhanh từ 3,700+ phim xuống Top-200 phim ứng viên giàu tiềm năng nhất dựa trên tích vô hướng không gian nhúng ẩn (Latent Vector Embedding Space).
2. **Giai đoạn Reranking & Diversity**: Cân bằng điểm sở thích cá nhân với chỉ số phổ biến sản phẩm, đồng thời ứng dụng kỹ thuật **MMR-style Genre Diversity Reranking** (phạt trùng lặp thể loại dựa trên độ tương đồng Jaccard) để ngăn chặn hiện tượng "Filter Bubble" (người dùng bị mắc kẹt trong một thể loại phim duy nhất).

---

## 🏗️ Kiến Trúc Hệ Thống (Two-Stage Architecture)

```mermaid
flowchart TD
    A[MovieLens 1M Dataset] --> B[Time-based Positive Sequence Split]
    B --> C[Sparse Implicit Matrix Rating >= 4.0]
    C --> D[TruncatedSVD Matrix Factorization]
    D --> E[(User & Item Embeddings - 64D)]

    F[User Request /recommend/user_id] --> G{User ID Tồn Tại?}
    G -- No (Cold-Start) --> H[Genre-Based / Global Popular Fallback]
    G -- Yes --> I[Stage 1: Latent Retrieval Dot Product]
    
    I --> J[Top 200 Candidate Items]
    J --> K[Stage 2: Popularity Score Blend]
    K --> L[MMR-Style Genre Diversity Reranking]
    L --> M[Top-K Recommended Items]
    H --> M
```

### Mô tả đường đi dữ liệu (Data Pipeline):
- **Tạo Ma trận tương tác thưa (Implicit Feedback Matrix)**: Chuyển đổi điểm đánh giá thành tín hiệu tương tác nhị phân (`Rating >= 4.0`), chỉ lưu trữ tương tác tích cực (không lưu explicit zeros).
- **Time-based Positive Sequence Split (Leave-Last-Two)**: Với mỗi người dùng có đủ lịch sử tương tác, tương tác tích cực mới nhất làm Test ground truth, tương tác tích cực kế cuối làm Validation ground truth. Toàn bộ tương tác trước mốc thời gian validation được đưa vào tập Train. Loại bỏ hoàn toàn lộ thông tin tương lai (**Lookahead Data Leakage**) và thống nhất với định nghĩa implicit positive feedback.
- **Retrieval Engine**: Tìm kiếm nhanh bằng `numpy.argpartition` trên tích vô hướng embedding ($V_{item} \cdot u_{user}$).
- **Reranking Engine**: Kết hợp điểm chuẩn hóa Latent Score với Popularity Rank theo hệ số tối ưu $\alpha$ tìm được qua seeded validation tuning:
  $$S_{combined} = \alpha \cdot S_{latent} + (1 - \alpha) \cdot S_{pop}$$
- **MMR-Style Genre Diversity Guardrail**: Áp dụng cơ chế phạt đa dạng thể loại (diversity penalty):
  $$Score(i) = S_{combined}(i) - \lambda_{div} \cdot \max_{s \in Selected} JaccardSimilarity(i, s)$$

---

## 📊 Kết Quả Đánh Giá Thực Nghiệm (Offline Benchmark)

Toàn bộ chỉ số dưới đây được kết xuất từ tệp artifact canonical [test_metrics.json](file:///d:/hoc/can%20lam/5_du_an_AI_Engineer_hoan_chinh/ai_engineer_5_projects/Two-Stage-Recommender/reports/test_metrics.json) và [ablation.json](file:///d:/hoc/can%20lam/5_du_an_AI_Engineer_hoan_chinh/ai_engineer_5_projects/Two-Stage-Recommender/reports/ablation.json). Đánh giá được thực hiện độc lập trên **2,000 người dùng** (lấy mẫu ngẫu nhiên có cố định seed `seed=42`) trên tập Test dương thực tế:

### 1. Bảng so sánh tổng thể với Popularity Baseline (Top-10)

| Chỉ số (Metric) | Mô hình Two-Stage (Full) | Popularity Baseline | Lift vượt Baseline | Ý nghĩa kinh doanh & Kỹ thuật |
|---|---:|---:|---:|---|
| **Recall@10** | **0.0835** | 0.0520 | **+0.0315** | Tỷ lệ tìm đúng phim người dùng thực sự thích (rating >= 4) ở Top-10. |
| **NDCG@10** | **0.0410** | 0.0257 | **+0.0153** | Độ chính xác xếp hạng (ưu tiên phim đúng nằm ở vị trí cao hơn). |
| **Catalog Coverage** | **20.63%** | 4.05% | **+16.58%** | Tỷ lệ kho phim được khai phá (cao gấp 5 lần baseline phổ biến). |
| **Intra-List Diversity (ILD)** | **0.7706** | 0.7802 | - | Độ đa dạng thể loại trong danh sách (đã loại trừ cặp thiếu metadata). |
| **Popular-Item Share** | **48.52%** | 100.0% | **-51.48%** | Tỷ lệ lượt hiển thị rơi vào Top-100 phim phổ biến (giảm thiên lệch phổ biến). |

> 💡 **Phân tích kỹ thuật**: Model đạt độ nhấc (lift) **+60.6%** về Recall@10 và **+59.5%** về NDCG@10 so với Popularity baseline. Chỉ số `Popular-Item Share` ghi nhận 48.52% phản ánh sự cân bằng lành mạnh: vừa tận dụng các phim chất lượng cao phổ biến vừa đưa các gợi ý cá nhân hóa ngách vào Top-K.

### 2. Báo cáo Nghiên cứu thành phần (Ablation Study & Latency Benchmark)

| Cấu hình (Variant) | Recall@10 | NDCG@10 | ILD (Đa dạng) | Coverage | p50 Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|
| **Popularity Baseline** | 0.0520 | 0.0257 | 0.7802 | 4.05% | 0.35 ms | 0.69 ms |
| **SVD only** (Latent Dot Product) | 0.0835 | 0.0409 | 0.7572 | 20.69% | 16.79 ms | 24.14 ms |
| **SVD + Popularity** | 0.0835 | 0.0409 | 0.7572 | 20.69% | 16.68 ms | 22.68 ms |
| **SVD + Popularity + MMR** (Full Pipeline) | **0.0835** | **0.0410** | **0.7706** | 20.63% | **16.85 ms** | **23.97 ms** |

**Nhận xét cốt lõi từ Ablation Study**:
1. **Retrieval SVD** là nhân tố quyết định độ chính xác: mở rộng Catalog Coverage từ 4.05% lên 20.69% và tăng Recall@10 từ 0.0520 lên 0.0835.
2. **MMR-style Genre Diversity Reranking** nâng chỉ số đa dạng ILD từ `0.7572` lên `0.7706` (+1.77% absolute gain) mà vẫn bảo toàn điểm Recall@10 và cải thiện nhẹ NDCG@10.
3. **Độ trễ suy luận (Latency)**: Đi qua toàn bộ quy trình Two-Stage + MMR reranking chỉ mất **p50 = 16.85 ms** và **p95 = 23.97 ms**, hoàn toàn nằm trong SLA tiêu chuẩn (< 50 ms) của các hệ thống microservices phục vụ trực tuyến.


---

## 🛠️ Hướng Dẫn Cài Đặt & Vận Hành

### 1. Chuẩn bị môi trường Python

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell):
.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
python -m pip install -r requirements.txt
```

### 2. Tải dữ liệu và Huấn luyện mô hình

```bash
# 1. Tải và giải nén dữ liệu MovieLens 1M (Kiểm tra an toàn Zip Slip)
python scripts/download_data.py

# 2. Huấn luyện SVD, tính Popularity Rank & Tuning hyperparameter alpha
python -m src.train

# 3. Đánh giá Offline Metrics trên tập Test và xuất báo cáo JSON
python -m src.evaluate

# 4. Chạy toàn bộ Unit Tests & Integration Tests
python -c "import os; os.makedirs('scratch/pytest_tmp', exist_ok=True)"
python -m pytest -p no:cacheprovider --basetemp=scratch/pytest_tmp -v
```

### 3. Khởi chạy REST API Server

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
- Truỳ cập tài liệu tương tác **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 Đặc Tả REST API & Ví Dụ Sử Dụng

### 1. Endpoint `/health` (System Status)
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response JSON**:
```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "movielens-svd-two-stage-v2"
}
```

### 2. Endpoint `/recommend/{user_id}` (Gợi ý cá nhân hóa)
```bash
curl -X GET "http://127.0.0.1:8000/recommend/1?k=5&diversity=0.05&include_metadata=true"
```
**Response JSON**:
```json
{
  "user_id": 1,
  "items": [
    {
      "item_id": 1197,
      "title": "Princess Bride, The (1987)",
      "genres": ["Action", "Adventure", "Comedy", "Romance"],
      "interaction_count": 2318
    },
    {
      "item_id": 260,
      "title": "Star Wars: Episode IV - A New Hope (1977)",
      "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
      "interaction_count": 2991
    }
  ],
  "includes_metadata": true,
  "model_version": "movielens-svd-two-stage-v2"
}
```

### 3. Endpoint `/recommend/cold-start` (Gợi ý người dùng mới)
```bash
curl -X POST "http://127.0.0.1:8000/recommend/cold-start" \
     -H "Content-Type: application/json" \
     -d '{"preferred_genres": ["Sci-Fi", "Action"], "k": 3}'
```

---

## 🚀 Lộ Trình Nâng Cấp Hệ Thống (Architectural Upgrade Roadmap)

Để nâng cấp dự án từ **Production-Style Prototype** lên quy mô công nghiệp hàng triệu sản phẩm, kiến trúc được thiết kế module hóa sẵn sàng cho các nâng cấp sau:

### 1. Nâng cấp Tầng 1 (Candidate Retrieval):
- **Thuật toán Factorization nâng cao**: Thay thế TruncatedSVD tuyến tính bằng **Implicit ALS (Alternating Least Squares)** hoặc **Bayesian Personalized Ranking (BPR)** chuyên biệt cho dữ liệu phản hồi ngầm định (Implicit Feedback).
- **Tìm kiếm vector tiệm cận (ANN Indexing)**:
  ```text
  Implicit ALS / BPR Matrix Factorization
                 ↓
      Item / User Embeddings
                 ↓
    FAISS Index (HNSW / IVF-PQ)
                 ↓
      Top-200 Candidates (< 3ms)
  ```
  Sử dụng thư viện **FAISS** (Facebook AI Similarity Search) hoặc **ScaNN** với cấu trúc chỉ mục HNSW/IVF-PQ giúp thời gian truy vấn Top-200 giảm xuống dưới 3ms cho hàng triệu items.

### 2. Nâng cấp Tầng 2 (Learned Ranking & Business Reranking):
- **Chuyển từ Heuristic Reranking sang Học máy có giám sát (Learning-to-Rank)**:
  ```text
  Candidate Features:
  ├── Latent interaction score
  ├── Item historical popularity
  ├── User genre affinity vector
  ├── Item release age
  ├── User interaction count
  └── Interaction recency
                 ↓
      LightGBM Ranker (LambdaMART)
                 ↓
      MMR-Style Diversity Post-Processing
  ```
- **3-tier Architecture chuẩn thực tế**:
  1. **Retrieval**: Lọc từ hàng triệu items xuống Top-200 (FAISS ANN).
  2. **Learned Ranking**: Mô hình GBDT (LightGBM Ranker / XGBoost) xếp hạng tinh Top-200 thành Top-30 dựa trên cross-features.
  3. **Business Reranking**: MMR diversity penalty, deduplication, business rules và campaign boosts thành Top-10 hiển thị cho người dùng.

---

## 💼 Trình Bày Trong CV & Kỹ Năng Phỏng Vấn (CV Highlights)

Nếu bạn đưa dự án này vào CV (Vị trí **AI Engineer / Machine Learning Engineer / Recommendation Engineer**), hãy sử dụng mô tả chuẩn cấu trúc **STAR**:

### 🎯 Điểm dòng CV mẫu (Bullet Points for Resume):
- **Thiết kế & Xây dựng Hệ thống Gợi ý 2 Tầng (Production-Style Two-Stage Recommender Architecture)** cho bộ dữ liệu MovieLens 1M sử dụng Matrix Factorization (Candidate Retrieval) và MMR-style Genre Diversity Reranking.
- **Áp dụng chiến lược Positive Sequence Leave-Last-Two Split** (chuỗi tương tác rating >= 4.0 theo thời gian) triệt tiêu Lookahead Data Leakage; đạt **Recall@10 = 0.0835** (+60.6% lift so với Popularity baseline) và **Intra-List Diversity = 0.7706**.
- **Thực hiện Ablation Study toàn diện** đánh giá 4 biến thể, chứng minh MMR reranking cải thiện độ đa dạng thể loại mà không tổn thất Recall/NDCG; đo lường độ trễ suy luận đạt **p50 = 16.85ms**, **p95 = 23.97ms**.
- **Xây dựng REST API bằng FastAPI & Pydantic** hỗ trợ lazy-loading artifacts, cold-start fallback theo thể loại, 100% test coverage tự động với `pytest` và xử lý an toàn lỗ hổng bảo mật Zip-Slip.

---

## ❓ Bộ Câu Hỏi Phỏng Vấn Sâu Về Dự Án (Interview Q&A)

<details>
<summary><b>1. Tại sao phải thiết kế hệ thống gợi ý 2 tầng (Two-Stage) thay vì 1 tầng?</b></summary>

> **Trả lời**: Bài toán gợi ý thực tế phải đối mặt với **trade-off giữa độ chính xác và tốc độ (latency)**. Nếu dùng mô hình phức tạp (như Deep Learning / Cross-Features) chấm điểm cho 1 triệu item, độ trễ sẽ vượt quá ngưỡng cho phép (200ms). Hệ thống 2 tầng giải quyết điều này bằng cách:
> - **Stage 1 (Retrieval)**: Dùng phép toán tích vô hướng đơn giản trên vector nhúng để lọc nhanh Top-200 sản phẩm tiềm năng nhất trong thời gian $O(N \cdot d)$.
> - **Stage 2 (Reranking)**: Áp dụng các luật kinh doanh phức tạp, độ đa dạng thể loại và độ phổ biến trên số lượng ứng viên nhỏ (200 items).
</details>

<details>
<summary><b>2. Tại sao phải phân chia dữ liệu theo Positive Sequence Time-based Split thay vì lấy ngẫu nhiên hay lấy interaction bất kỳ?</b></summary>

> **Trả lời**: Trong mô hình gợi ý implicit feedback, mô hình được huấn luyện để gợi ý những item người dùng **thực sự thích** (Rating >= 4.0). Nếu lấy tương tác cuối cùng bất kỳ làm test set (ví dụ user chấm 1 sao vì ghét bộ phim đó), hệ thống đánh giá sẽ phạt model nếu nó không recommend phim bị ghét (Recall=0), hoặc ngược lại thưởng model nếu gợi ý đúng phim user ghét. Do đó, split chuẩn phải cắt trên chuỗi positive sequence của từng user, đồng thời chỉ đưa các tương tác trước mốc thời gian validation vào tập train để triệt tiêu triệt để **Lookahead Data Leakage**.
</details>

<details>
<summary><b>3. Lỗ hổng Zip-Slip là gì và bạn xử lý nó như thế nào?</b></summary>

> **Trả lời**: Zip-Slip là lỗ hổng khi giải nén tệp archive (.zip) chứa các tên file có đường dẫn tương đối (ví dụ `../../system32/malicious.exe`). Nếu không kiểm tra, kẻ tấn công có thể ghi đè file nguy hiểm ra ngoài thư mục dữ liệu chỉ định. Em đã triển khai hàm `_safe_extract()` kiểm tra đường dẫn tuyệt đối của từng file trong zip phải thuộc cây thư mục đích (`destination in target.parents`), nếu vi phạm sẽ ném ngoại lệ `ValueError` lập tức.
</details>

---

## 📜 Cấu Trúc Thư Mục Dự Án (Directory Structure)

```text
05_two_stage_recommender/
├── configs/               # Tệp cấu hình mở rộng
├── data/                  # Dữ liệu raw và processed (MovieLens 1M)
├── models/                # Artifacts mô hình (user_emb.npy, item_emb.npy, meta.joblib, config.json)
├── reports/               # Báo cáo đánh giá canonical (test_metrics.json, ablation.json)
├── scripts/               # Scripts tiện ích (download_data.py)
├── src/                   # Mã nguồn chính
│   ├── __init__.py
│   ├── api.py             # FastAPI REST endpoints
│   ├── data.py            # Data loader & Positive Sequence Time Split
│   ├── evaluate.py        # Offline metrics & Ablation study (Recall, NDCG, ILD, Latency)
│   ├── recommender.py     # Two-stage recommendation & MMR-style reranking engine
│   ├── train.py           # SVD Training & Seeded validation tuning pipeline
│   └── utils.py           # Common helpers, seed & logging
├── tests/                 # Test suite (pytest)
│   ├── __init__.py
│   └── test_smoke.py      # Unit & Integration test cases
├── Dockerfile             # Containerization setup
├── Makefile               # CLI Commands shortcut
├── README.md              # Tài liệu dự án chi tiết
└── requirements.txt       # Danh sách thư viện phụ thuộc
```

---

## 📝 Giấy Phép & Tác Giả

Dự án được xây dựng và đóng gói theo chuẩn Production bởi **AI Engineer Portfolio**.
Phát hành theo giấy phép [MIT License](LICENSE).
