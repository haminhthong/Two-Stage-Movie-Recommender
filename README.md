# 🎬 Personalized Two-Stage Movie Recommendation Platform (MovieLens 1M)

> **Hệ thống gợi ý phim cá nhân hóa theo luồng:** historical events → point-in-time user state → SVD + Popularity + Genre retrieval → RRF top-200 → XGBRanker theo query group → relevance pool 40 → diversity guardrail → top-10. Khi ranker không vượt quality gate trên Dev, hệ thống fallback về retrieval order.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/pytest-automated-blue.svg)](tests/)

---

## Contract hiện tại

- Candidate contract duy nhất: `total_k=200`, gồm SVD 150, popularity 50, genre 50; hợp nhất bằng RRF với `rrf_k=60`.
- Nhãn `y=1`: positive interaction (`rating >= 4`). Nhãn `y=0`: candidate không trùng target, chỉ là sampled non-interaction proxy vì MovieLens không có impression logs.
- Ranker mới dùng `XGBRanker(objective="rank:ndcg")` và `groups`; target bị Stage 1 miss sẽ bị loại khỏi rank-training group, không được chèn nhân tạo.
- Score feature của từng source được giữ riêng; feature snapshot và seen filter tuân thủ `timestamp < as_of`.
- Ablation/tuning chỉ chạy trên Dev. Locked Test chỉ chạy một lần với pipeline đã freeze; báo cáo cũ trong `reports/` chỉ có giá trị chẩn đoán lịch sử.

## 📌 1. Bài Toán & Lý Do Cần Kiến Trúc Hai Tầng (Why Two-Stage?)

Trong các nền tảng lớn, retrieval trước giúp ranker chỉ xử lý candidate pool nhỏ. Project đặt **target latency budget <100ms**, nhưng không tuyên bố đạt SLA khi chưa benchmark lại; p95 hiện tại phải được đọc từ report của từng release.

Kiến trúc **Two-Stage Recommender** (dựa trên thiết kế kinh điển của Covington et al., Google/YouTube RecSys 2016) giải quyết triệt để sự đánh đổi giữa **độ chính xác (accuracy)**, **tính đa dạng (diversity)** và **tốc độ suy luận (latency)**:
- **Tầng 1 — Retrieval**: Chịu trách nhiệm *"Item đúng có lọt vào Top-200 ứng viên không?"*
- **Tầng 2 — Ranking**: Chịu trách nhiệm *"Nếu item đúng đã lọt vào candidates, mô hình có xếp nó lên Top-10 không?"*
- **Tầng 3 — Post-Ranking**: Chịu trách nhiệm *"Top-10 đề xuất có bị bẫy lọc (filter bubble), trùng lặp hay vi phạm ràng buộc nghiệp vụ không?"*

---

## 🏗️ 2. Kiến Trúc Hệ Thống Chuẩn (Target Production Architecture)

```text
                 USER–ITEM EVENTS
               MovieLens / event logs
                       │
                       ▼
┌────────────────────────────────────────────┐
│ 1. DATA PIPELINE                          │
│ • Schema validation (DataContract)         │
│ • Event normalization & SHA256 manifest    │
│ • Implicit Positive Contract (Rating >= 4) │
│ • Positive-Only User Genre Profiles        │
│ • Sampled Non-Interaction Negative Proxy   │
└───────────────────┬────────────────────────┘
                    ▼
┌────────────────────────────────────────────┐
│ 2. TEMPORAL 4-WAY SPLIT (Per-User Holdout) │
│                                            │
│ • Retrieval Train (t < T_rank)             │
│ • Rank Train Target (T_rank, 3rd-last)     │
│ • Validation Target (T_val, 2nd-last)      │
│ • Test Target (T_test, latest positive)    │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│ 3. STAGE 1 — MULTI-SOURCE RETRIEVAL       │
│ ┌───────────────┬────────────┬───────────┐ │
│ │ SVD Embedding │ Popularity │   Genre   │ │
│ │  (Top 150)    │  (Top 50)  │  (Top 50) │ │
│ └───────┬───────┴─────┬──────┴─────┬─────┘ │
│         └─────────────┼────────────┘       │
│                       ▼                    │
│   Candidate Merger & Deduplication (~200)  │
│   Rich Candidate Metadata (source, ranks)  │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ 4. STAGE 2 — GROUP-AWARE XGBRANKER        │
│ • User Features (counts, avg_rating, etc.) │
│ • Item Features (pop, counts, genres)      │
│ • User x Item (latent, genre affinity,     │
│   percentile, retrieval ranks)             │
│                       ↓                    │
│   Supervised Pointwise Scorer              │
│   Top 50 Ranked Candidates                 │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ 5. POST-RANKING (Reranking & Constraints)  │
│ • Seen Items Hard Filter (-inf)            │
│ • Bitmask-Optimized MMR (rerank_pool_k=40) │
│ • Catalog Coverage & Long-Tail Calibration │
│                       ↓                    │
│   Top-K Final Recommendations (K=10)       │
└──────────────┬─────────────────────────────┘
               ▼
      ┌─────────────────┐
      │                 │
      ▼                 ▼
 Offline Evaluation    Online Serving API
 Full-Funnel Metrics   Versioned Bundle Load
 Latency Benchmarks    FastAPI (/recommend)
 Long-Tail Exposure    Fresh Session Signals
```

---

## ⏱️ 3. Giao Thức Dữ Liệu & Kiểm Soát Rò Rỉ Thời Gian (Temporal Data Contracts)

### 3.1. Phân định ranh giới Tín hiệu huấn luyện vs Bộ lọc đã xem (Signal vs Filter)
- **Tín hiệu Huấn luyện (Training Signal)**: Bài toán implicit feedback coi tương tác có $\text{Rating} \ge 4.0$ là **tương tác tích cực ngầm định (implicit positive)**. Các tương tác $\text{Rating} < 4.0$ không tham gia tạo liên kết trong ma trận nhị phân.
- **Sửa lỗi Genre Profile (P0.1)**: Phiên bản trước gộp toàn bộ rating (kể cả 1★, 2★) vào hồ sơ thể loại của người dùng, dẫn đến nghịch lý người dùng chấm phim kinh dị 1 sao nhưng hệ thống vẫn coi kinh dị là thể loại yêu thích. Ở phiên bản hiện tại, **User Genre Profiles được trích xuất nghiêm ngặt chỉ từ tương tác tích cực ($\text{Rating} \ge 4.0$)**.
- **Bộ lọc Sản phẩm Đã xem (Seen Filter Guardrail)**: Bất kỳ sản phẩm nào người dùng đã từng tương tác trong quá khứ (kể cả phim chấm 1 sao) đều bị loại bỏ khi suy luận nhằm tránh lãng phí vị trí Top-K.

### 3.2. Phân chia 4 nhánh thời gian (Per-User Temporal 4-Way Split)
Để huấn luyện mô hình Learned Ranker độc lập mà không làm nhiễm độc tập Validation và Test, hệ thống nâng cấp điều kiện $\text{min\_positives} = 4$ và chia trục thời gian của từng user thành 4 mốc rõ rệt:

```text
User Positive Timeline:
... earlier positives ───► [Rank-Train Target] ───► [Validation Target] ───► [Test Target]
        │                           │                       │                   │
  Retrieval Train           Train Ranker Features      Tune MMR Lambda      REPORT ONLY
  (Build Embeddings)       (Sample Negatives Proxy)   (Pool-k Evaluation)  (Zero Tuning)
```

1. **Retrieval Train**: Toàn bộ lịch sử trước mốc $T_{rank}$. Dùng để xây dựng ma trận thưa và huấn luyện ma trận nhúng TruncatedSVD.
2. **Rank Train Target**: Tương tác tích cực thứ 3 từ dưới lên ($T_{rank}$). Dùng làm nhãn $y=1$; các ứng viên được sinh ra từ lịch sử quá khứ nhưng không phải target đóng vai trò **sampled non-interaction negative proxy** ($y=0$).
3. **Validation Target**: Tương tác kế cuối ($T_{val}$). Dùng để tune siêu tham số đa dạng hóa MMR, ngưỡng ranking và lựa chọn kiến trúc ranker.
4. **Test Target**: Tương tác mới nhất ($T_{test}$). Giữ nguyên vẹn cho báo cáo chính thức.

> [!NOTE]
> **Khuyến cáo về định nghĩa Split**: Đây là giao thức **per-user temporal holdout**, đảm bảo lịch sử quá khứ của một user không chứa tương lai của chính họ. Không nên gọi đây là *strict global point-in-time simulation* vì timestamp của user A ở tập test có thể nhỏ hơn timestamp của user B ở tập train.

> [!WARNING]
> **Bản chất nhãn Negative**: MovieLens không có nhật ký phơi nhiễm (impression logs). Nhãn 0 trong Rank-Train là **sampled non-interaction proxy** (sản phẩm được ứng viên gợi ý nhưng user chưa tương tác trong bước đó), không phải bằng chứng user đã nhìn thấy và từ chối.

---

## 🎯 4. Tầng 1: Multi-Source Candidate Retrieval Engine

Thay vì chỉ dựa vào một nguồn duy nhất, hệ thống kết hợp 3 kênh trích xuất ứng viên độc lập thông qua `MultiSourceRetriever`:

1. **SVD Collaborative Retriever (Top 150)**: Phân rã ma trận ẩn TruncatedSVD ($d=64$), nắm bắt tương quan tiềm ẩn giữa người dùng và sản phẩm qua tích vô hướng vector.
2. **Popularity Retriever (Top 50)**: Khai thác tiên nghiệm toàn cục bằng độ phổ biến $\log(1 + \text{count})$, đảm bảo độ bao phủ các item thịnh hành.
3. **Genre Affinity Retriever (Top 50)**: Dựa trên phân phối thể loại tích cực trong quá khứ của người dùng, truy xuất các phim phù hợp sở thích thể loại.

### Cấu trúc dữ liệu Candidate giàu thông tin:
```python
@dataclass(frozen=True)
class Candidate:
    item_id: int
    retrieval_score: float
    retrieval_source: str                 # "svd", "popularity", "genre", "multi_source"
    retrieval_rank: int = 0               # Thứ hạng trong kênh trích xuất
    source_scores: dict[str, float] = field(default_factory=dict)
```

---

## ⚖️ 5. Tầng 2: Supervised Learned Ranker (Learning-to-Rank)

Thay thế phép tổng hợp tuyến tính cố định bằng `XGBRanker(objective="rank:ndcg")` học theo query/user group. `WeightedFusionRanker` chỉ giữ làm baseline và fallback khi ranker không qua Dev gate.

### 19 đặc trưng source-specific (Hierarchical Feature Store):
| Nhóm Đặc Trưng | Tên Đặc Trưng | Ý Nghĩa Kỹ Thuật |
|---|---|---|
| **User Features** | `user_interaction_count` | Tổng số lượt tương tác trong lịch sử |
| | `user_positive_count` | Số tương tác tích cực ($\ge 4.0$) |
| | `user_avg_rating` | Điểm đánh giá trung bình của user |
| | `user_genre_entropy` | Độ phân tán sở thích thể loại |
| **Item Features** | `item_popularity` | Log-transformed popularity prior $\log(1+\text{count})/\log(1+\max)$ |
| | `item_avg_rating` | Điểm đánh giá trung bình của phim |
| | `item_rating_count` | Tổng số lượt đánh giá phim nhận được |
| | `item_genre_count` | Số lượng thể loại của bộ phim |
| | `item_popularity_percentile` | Thứ hạng bách phân vị về độ phổ biến |
| **User $\times$ Item** | `latent_score` | Điểm tích vô hướng SVD chuẩn hóa min-max cục bộ |
| | `raw_latent_score` | Điểm tích vô hướng SVD nguyên bản (giữ thang đo tuyệt đối) |
| | `genre_affinity` | Độ tương hợp Jaccard giữa user positive genres và item genres |
| | `user_item_genre_overlap` | Số thể loại trùng lặp tuyệt đối |
| | `retrieval_rank` | Thứ hạng trích xuất ở Tầng 1 |
| | `retrieval_rank_percentile` | Tỷ lệ phần trăm thứ hạng trong danh sách ứng viên |

---

## 🎨 6. Tầng 3: Post-Ranking & Tối Ưu Hóa Đa Dạng Hóa (MMR Diversity)

### 6.1. Đồng nhất Pipeline và giải quyết Training-Serving Skew (P0.2)
- Phiên bản cũ tune diversity với `pool_size = 40` nhưng khi phục vụ online lại chạy MMR trên toàn bộ 200 candidates.
- Hệ thống chuẩn hóa cấu hình bất biến trên toàn bộ Train, Val, Test và Serving:
  $$\text{candidate\_k} = 200 \longrightarrow \text{ranking\_k} = 50 \longrightarrow \text{rerank\_pool\_k} = 40 \longrightarrow \text{final\_k} = 10$$

### 6.2. Bản chất "MMR Candidate-Pool Approximation" (P0.8)
Việc giới hạn pool rerank ở Top 40 là một phép xấp xỉ thực nghiệm (**Candidate-Pool Approximation**), không phải bảo chứng toán học. Thực nghiệm chứng minh:
- `pool_k = 40` đạt Recall@10 **cao hơn** `pool_k = 200` (0.043 vs 0.039) do không bị phạt đa dạng quá đà.
- Tiết kiệm hơn **2.5x độ trễ** so với chạy MMR trên 200 ứng viên.

### 6.3. Tối ưu hóa Bitmask & POPCNT (Bit-Count)
Khoảng cách Jaccard thể loại được chuyển đổi sang số nguyên 32-bit (Bitmask):
```python
# Phép toán tập hợp O(N) chuyển thành thao tác vi xử lý 1 chu kỳ CPU:
intersection = (mask_a & mask_b).bit_count()
union = (mask_a | mask_b).bit_count()
jaccard = intersection / union
```
Các con số latency ở report cũ chỉ là benchmark tham khảo; cần đo lại trên từng release và workload thực tế, không suy ra SLA từ benchmark này.

---

## 📊 7. Kết Quả Đánh Giá Thực Nghiệm (Offline Benchmark)

Các số liệu trong [reports/test_metrics.json](reports/test_metrics.json) và [reports/ablation.json](reports/ablation.json) là report lịch sử. Ablation cũ từng dùng Test nên không được dùng để chọn model hoặc quảng bá champion; report mới phải ghi rõ Dev/Locked Test protocol:

### 7.1. Phễu Đánh Giá Từng Tầng (Funnel Stage Metrics — report lịch sử)
| Giai Đoạn (Stage) | Chỉ Số Đo Lường | Giá Trị | Ý Nghĩa Kỹ Thuật |
|---|---|---:|---|
| **Catalog Availability** | `target_in_catalog_rate` | **99.98%** | 6,034 / 6,035 target items có mặt trong catalog huấn luyện. |
| | `cold_item_test_share` | **0.016%** | Chỉ 1 item duy nhất bị cold-start ở tập test. |
| **Stage 1: Retrieval** | `candidate_recall@50` | **25.58%** | 1/4 target items nằm trong top 50 ứng viên đầu tiên. |
| | `candidate_recall@100` | **37.65%** | Hơn 37% target items nằm trong top 100 ứng viên. |
| | `candidate_recall@200` | **52.48%** | Hơn một nửa ground-truth items lọt qua phễu Tầng 1. |
| **Stage 2: Ranking** | `ranker_recall@10` | **0.0557** | Tỷ lệ đưa target item vào Top 10 sau mô hình XGBRanker (số liệu lịch sử). |
| | `ranker_ndcg@10` | **0.0239** | Thứ hạng tương đối có trọng số vị trí. |
| **Stage 3: Final Post-MMR** | `final_recall@10` | **0.0557** | Không làm sụt giảm Recall sau bước lọc đa dạng thể loại. |

---

### 7.2. So Sánh Mô Hình Two-Stage vs Popularity Baseline
| Chỉ Số Đánh Giá | Learned Two-Stage | Popularity Baseline | Absolute Gain | Relative Lift | Ý Nghĩa Nghiệp Vụ |
|---|---:|---:|---:|---:|---|
| **Recall@10 / HitRate@10** | **0.0557** | 0.0444 | **+0.0113** | **+25.37%** | Tìm đúng phim thích ở Top-10 (*Leave-one-out*). |
| **NDCG@10** | **0.0239** | 0.0215 | **+0.0024** | **+11.13%** | Ưu tiên xếp phim đúng ở các vị trí đầu danh sách. |
| **MRR@10** | **0.0143** | 0.0147 | -0.0004 | -2.64% | Nghịch đảo vị trí xuất hiện đầu tiên của item đúng. |
| **Catalog Coverage** | **48.34%** | 4.05% | **+44.29%** | **+1,093%** | **Gấp gần 12 lần baseline! Phủ gần 50% kho phim.** |
| **Intra-List Diversity (ILD)**| **0.7358** | 0.7832 | -0.0474 | -6.05% | Độ đa dạng thể loại trong danh sách Top-10. |
| **Novelty@10** | **10.6113** | 8.1283 | **+2.4830** | **+30.55%** | Khám phá phim mới lạ, vượt xa baseline phổ biến. |

---

### 7.3. Giải Quyết Thiên Lệch Phổ Biến & Khám Phá Long-Tail (Debiasing Breakthrough)
Các con số dưới đây là diagnostic của phiên bản cũ; không dùng để chọn champion cho release mới:

```text
Phiên bản cũ (Heuristic Fusion):
████████████████████████████████████████ 89.8% Head Exposure (Top 10% phim hot)
████ 10.2% Mid Exposure
 0.0% Tail Exposure (Hoàn toàn bị bỏ rơi)

Phiên bản mới (Learned Two-Stage Platform):
██████████████ 36.0% Head Exposure (Giảm mạnh tập trung độc quyền phim hot)
█████████████████████████ 63.4% Mid Exposure (Khai phóng danh mục tiềm năng!)
█ 0.59% Tail Exposure (Bắt đầu phân phối phơi nhiễm cho phim ngách)
```

👉 **Catalog Coverage tăng vọt từ 23.01% lên 48.34%**, giúp hệ thống trở thành một nền tảng khám phá nội dung thực sự thay vì chỉ là cỗ máy khuếch đại phim bom tấn.

---

### 7.4. Nghiên Cứu Ablation & Khảo Sát MMR Candidate-Pool Size (report lịch sử)
Bảng dưới đây là benchmark cũ để tham khảo quan hệ đánh đổi giữa Recall, ILD và độ trễ; ablation hiện tại phải chạy trên Dev:

| Cấu hình Thử Nghiệm | Recall@10 | Intra-List Diversity (ILD) | Độ Trễ p50 | Độ Trễ p95 |
|---|---:|---:|---:|---:|
| **Popularity Baseline** | 0.0390 | 0.7832 | 1.24 ms | 5.36 ms |
| **SVD Only (Latent Retrieval)** | 0.0850 | 0.7573 | 16.66 ms | 25.79 ms |
| **SVD + Popularity (Weighted Baseline)** | 0.0840 | 0.7600 | 15.94 ms | 29.36 ms |
| **Stage-2 Learned Ranker (XGBRanker)** | 0.0500 | 0.7320 | 61.79 ms | 279.11 ms |
| **Full Pipeline (Learned + MMR Pool 40)** | **0.0430** | **0.8740** | **48.27 ms** | **170.17 ms** |
| • *Ablation: MMR Pool 20* | 0.0410 | 0.8280 | 42.56 ms | 151.81 ms |
| • *Ablation: MMR Pool 40 (Optimal)* | **0.0430** | **0.8740** | **48.27 ms** | **170.17 ms** |
| • *Ablation: MMR Pool 100* | 0.0410 | 0.9124 | 100.60 ms | 403.28 ms |
| • *Ablation: MMR Pool 200* | 0.0390 | 0.9478 | 120.18 ms | 297.79 ms |

#### 💡 Phát hiện cốt lõi từ khảo sát:
1. Benchmark cũ từng cho thấy `MMR Pool 40` là một điểm cân bằng hợp lý; kết quả này không thay thế relevance guardrail của release mới.
2. Chi phí độ trễ phụ thuộc phần cứng, batch size và workload; `rerank_pool_k = 40` là contract hiện tại cần được kiểm chứng lại bằng benchmark của release.

---

## 📦 8. Quản Lý Phiên Bản Model & Nguồn Gốc Dữ Liệu (Artifact Lifecycle & Provenance)

Hệ thống quản lý artifacts theo chuẩn MLflow/Production Registry:
```text
models/
├── production.json                # Chỉ trỏ release đã qua promotion gate
└── candidates/                    # Release Candidate, chưa active
    └── v5-multisource-ranker/
    ├── config.json                # Hyperparameters, split protocol, feature names
    ├── data_manifest.json         # Raw files checksums (SHA256 ratings.dat, movies.dat)
    ├── temporal_split_manifest.json # Data split timestamps, user/item interaction counts
    ├── retrieval/
    │   ├── user_emb.npy           # SVD User Embeddings [6040, 64]
    │   └── item_emb.npy           # SVD Item Embeddings [3703, 64]
    ├── ranking/
    │   └── ranker.joblib          # Trained XGBoost Ranker & Feature Normalizers
    └── metadata/
        └── meta.joblib            # Titles, genres, bitmasks, seen_by_user, item popularity
```

---

## 🌐 9. Đặc Tả REST API & Hành Vi Tươi Mới (Fresh Session Signals)

### 9.1. Endpoint `/health` (Trạng thái hệ thống & Model Version)
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "v5-multisource-ranker"
}
```

### 9.2. Endpoint `/recommend/{user_id}` (Hỗ trợ hành vi phiên gần nhất `recent_items`)
Khi người dùng vừa tương tác trong phiên hiện tại (chưa kịp retrain embedding), tham số `recent_items` cho phép cập nhật tức thời bộ lọc seen và hồ sơ thể loại:
```bash
curl -X GET "http://127.0.0.1:8000/recommend/1?k=2&recent_items=260,1197&include_scores=true"
```
```json
{
  "user_id": 1,
  "strategy": "two_stage_learned_ranking",
  "items": [
    {
      "item_id": 1210,
      "title": "Star Wars: Episode VI - Return of the Jedi (1983)",
      "genres": ["Action", "Adventure", "Romance", "Sci-Fi", "War"],
      "interaction_count": 2883,
      "scores": {
        "retrieval": 0.8124,
        "popularity": 0.9652,
        "ranking": 0.7421,
        "diversity_penalty": 0.0,
        "final": 0.7421
      },
      "reason_codes": ["COLLABORATIVE_MATCH", "GENRE_MATCH"],
      "explanation": "Tín hiệu cộng tác và mức phù hợp thể loại; danh sách được kiểm soát độ trùng lặp thể loại."
    }
  ],
  "model_version": "v5-multisource-ranker",
  "latencies_ms": {
    "retrieval": 2.98,
    "ranking": 55.78,
    "diversity": 0.09,
    "total": 60.32
  }
}
```

### 9.3. Endpoint `/recommend/cold-start` (Phân biệt New User vs New Item)
Hỗ trợ cả người dùng mới hoàn toàn (qua sở thích thể loại) và sản phẩm mới vào kho catalog.
```bash
curl -X POST "http://127.0.0.1:8000/recommend/cold-start" \
     -H "Content-Type: application/json" \
     -d '{"preferred_genres": ["Animation", "Children'\''s"], "k": 3}'
```

---

## 💻 10. Hướng Dẫn Cài Đặt & Vận Hành (Reproducibility Guide)

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell (hoặc source .venv/bin/activate trên Linux)

# 2. Cài đặt thư viện phụ thuộc (scikit-learn, xgboost, fastapi, pytest, v.v.)
python -m pip install -r requirements.txt

# 3. Tải và kiểm tra checksum dữ liệu MovieLens 1M
python scripts/download_data.py

# 4. Huấn luyện pipeline (multi-source retrieval -> query-group ranker -> Release Candidate, chưa active)
python -m src.train

# 5. Đánh giá: funnel trên Locked Test; ablation/baseline chỉ trên Dev
python -m src.evaluate

# 6. Chạy kiểm thử tự động
python -m pytest tests/ -v

# 7. Khởi chạy microservice API
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
- Truy cập tương tác **Swagger UI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🗂️ 11. Cấu Trúc Mã Nguồn Chuẩn Hóa (Project Structure)

```text
Two-Stage-Recommender/
├── configs/                       # Cấu hình mở rộng
├── data/raw/ml-1m/                # Dữ liệu gốc MovieLens 1M (ratings.dat, movies.dat)
├── models/                        # Versioned Artifact Registry
│   ├── production.json            # Active version pointer sau promotion
│   └── candidates/                # RC chưa active; chỉ promote sau quality gates
│       └── v5-multisource-ranker/
│           ├── config.json        # Model hyperparameters & contracts
│           ├── data_manifest.json # Input files SHA256 checksums
│           ├── temporal_split_manifest.json # Temporal split statistics
│           ├── retrieval/         # user_emb.npy, item_emb.npy
│           ├── ranking/           # ranker.joblib (XGBRanker)
│           └── metadata/          # meta.joblib (titles, genres, seen)
├── reports/                       # Báo cáo đánh giá chính thức
│   ├── test_metrics.json          # Funnel Stage Metrics, Lift, Exposure, Latencies
│   └── ablation.json              # Baselines trên Dev, không grid trên Test
├── scripts/
│   └── download_data.py           # Tải MovieLens 1M kèm SHA256 validation
├── src/
│   ├── __init__.py
│   ├── api.py                     # FastAPI endpoints (with recent_items & debug scores)
│   ├── config.py                  # Dataclasses cấu hình bất biến
│   ├── train.py                   # Master offline training pipeline
│   ├── evaluate.py                # Funnel evaluator & pool ablation benchmark
│   ├── utils.py                   # Logging, seed control, manifests helpers
│   ├── data/                      # [Data Pipeline & Temporal Split]
│   │   ├── schema.py              # DataContract & validation
│   │   ├── loader.py              # Raw data loader & SHA256 verification
│   │   ├── interactions.py        # Positive filtering (>=4.0) & positive-only genre profiles
│   │   ├── split.py               # 4-way temporal split & backward-compatible time_split
│   │   └── manifest.py            # Data & split provenance manifests
│   ├── retrieval/                 # [Stage 1: Candidate Generation]
│   │   ├── base.py                # Candidate dataclass & CandidateRetriever ABC
│   │   ├── svd.py                 # SVDRetriever (TruncatedSVD dot product)
│   │   ├── popularity.py          # PopularityRetriever
│   │   ├── genre.py               # GenreRetriever (Positive profile based)
│   │   └── merger.py              # MultiSourceRetriever (dedupe & metadata enrich)
│   ├── ranking/                   # [Stage 2: Scoring & Learning-to-Rank]
│   │   ├── features.py            # 19 source-specific/PIT-safe features
│   │   ├── dataset.py             # RankDatasetBuilder (negative sampling proxy)
│   │   ├── model.py               # LearnedRanker (XGBoost/LR) & WeightedFusionRanker
│   │   ├── trainer.py             # Training loop & validation evaluation
│   │   └── scorer.py              # TwoStageRanker facade
│   ├── reranking/                 # [Stage 3: Post-Ranking Diversity & Rules]
│   │   ├── diversity.py           # DiversityReranker (Bitmask & bit_count optimized)
│   │   └── business_rules.py      # Hard seen filter & popularity calibrator
│   ├── evaluation/                # [Evaluation & Metrics]
│   │   ├── retrieval_metrics.py   # Candidate Recall@50/100/200 & catalog rates
│   │   ├── ranking_metrics.py     # NDCG, HitRate, MRR
│   │   ├── metrics.py             # ILD, Novelty, Exposure, Coverage
│   │   ├── latency.py             # Detailed stage-level latency timers
│   │   └── evaluator.py           # FullFunnelEvaluator
│   ├── artifacts/                 # [Artifact Management & Registry]
│   │   ├── schema.py              # Bundle dataclasses & version schema
│   │   ├── writer.py              # save_versioned_bundle
│   │   └── loader.py              # load_production_bundle
│   └── serving/                   # [Serving Engine & Cold-Start]
│       ├── cold_start.py          # ColdStartPolicy (New user & New item separation)
│       └── recommender.py         # TwoStageRecommenderEngine (with recent session signals)
├── tests/
│   ├── test_smoke.py              # Basic integration tests
│   ├── test_recommender.py        # Seen filter, temporal integrity, API tests
│   └── test_pipeline_integrity.py # 14 comprehensive tests (data contracts, no leaks, MMR pool)
├── Dockerfile                     # Production containerization
├── Makefile                       # Developer shortcuts
└── requirements.txt               # Dependencies specification
```

---

## 💼 12. Điểm Nhấn Phỏng Vấn Kỹ Thuật (STAR Interview Highlights)

### 🎯 Điểm dòng CV mẫu (Resume Bullet Points):
- **Thiết kế hệ thống gợi ý 2 tầng** trên MovieLens 1M: Stage 1 Multi-Source Retrieval (SVD + Popularity + Genre) với RRF, Stage 2 group-aware XGBRanker với 19 feature source-specific và Stage 3 diversity dưới relevance guardrail.
- **Thiết kế giao thức Per-User Temporal 4-Way Split** (`min_positives = 4`) tách biệt nghiêm ngặt: Retrieval Train $\rightarrow$ Rank Train $\rightarrow$ Validation $\rightarrow$ Test, triệt tiêu hoàn toàn rò rỉ dữ liệu tương lai (**Lookahead Data Leakage**) và loại bỏ Training-Serving Skew.
- **Sửa lỗi logic phân định tín hiệu (Data Contract)**: Chuẩn hóa User Genre Profile nghiêm ngặt chỉ dựa trên tương tác tích cực ($\ge 4.0$), loại bỏ hiện tượng thiên lệch sở thích tiêu cực.
- **Xây dựng phễu đo lường hiệu năng từng tầng (Funnel Stage Metrics)**: Tách riêng retrieval recall, ranker quality, final quality, conditional metrics và rescue metrics; mọi số liệu production phải lấy từ report của release cụ thể.
- **Đo lường đa dạng hóa & thiên lệch danh mục**: Theo dõi Catalog Coverage, ILD, Novelty và Head/Mid/Tail Exposure; không quảng bá các số liệu lịch sử trong `reports/` như kết quả của release mới.
- **Tối ưu hóa MMR bằng Bitmask**: Dùng toán tử bitwise và `bit_count()`; latency phải được benchmark theo từng release, không suy diễn SLA từ benchmark cũ.
- **Quản trị vòng đời Model Artifacts**: Train ghi Release Candidate; `manifest.json` kiểm tra hash/schema; chỉ `promote_release()` sau Dev gate và Locked Test mới cập nhật `production.json`. Loader fail-closed, không trộn artifact root/legacy.

---

## ❓ 13. Bộ Câu Hỏi Phỏng Vấn Chuyên Sâu (Deep-Dive Interview Q&A)

<details>
<summary><b>1. Tại sao giai đoạn Stage 1 cần Multi-Source Candidate Generation thay vì chỉ dùng duy nhất SVD?</b></summary>

> **Trả lời**: Mỗi thuật toán retrieval có một góc nhìn (inductive bias) khác nhau:
> - **Collaborative Filtering (SVD)** nắm bắt rất tốt các tương quan tiềm ẩn phức tạp giữa người dùng tương đồng, nhưng dễ bỏ sót các sản phẩm mới hoặc các sở thích thể loại chưa được phân rã rõ.
> - **Popularity Retriever** cung cấp điểm neo an toàn cho các sản phẩm thịnh hành đang có xác suất tiên nghiệm cao.
> - **Genre Affinity Retriever** trực tiếp đưa vào các sản phẩm đúng thể loại yêu thích gần đây của người dùng.
> Việc kết hợp nhiều nguồn trích xuất cung cấp một tập ứng viên đa dạng cho Stage-2 Learned Ranker. Candidate Recall phải được đọc từ funnel report của release đang đánh giá; số liệu lịch sử trong tài liệu này không phải quality gate hiện hành.
</details>

<details>
<summary><b>2. Tại sao nhãn negative trong tập Rank-Train lại gọi là "sampled non-interaction proxy" chứ không phải negative feedback thật?</b></summary>

> **Trả lời**: MovieLens chỉ ghi nhận các lượt đánh giá (Rating) từ người dùng; không ghi nhận nhật ký hiển thị (Impression Logs). Do đó, một bộ phim người dùng không đánh giá có thể vì hai lý do: (1) Họ đã thấy nhưng không thích (true negative), hoặc (2) Hệ thống chưa từng hiển thị phim đó cho họ (unobserved positive). Khi lấy các ứng viên trích xuất từ Tầng 1 mà không trùng với target item làm nhãn $y=0$, ta chỉ đang xấp xỉ bằng **Sampled Non-Interaction Proxy**. Việc trung thực với bản chất dữ liệu thể hiện sự am hiểu sâu sắc về Selection Bias trong RecSys công nghiệp.
</details>

<details>
<summary><b>3. Tại sao chạy MMR trên toàn bộ 200 candidates lại cho kết quả Recall thấp hơn chạy trên pool 40?</b></summary>

> **Trả lời**: Giải thuật MMR cân bằng giữa điểm liên quan ($S_{rel}$) và độ phạt tương đồng ($\lambda \cdot \text{Sim}$). Khi mở rộng pool lên 200, các sản phẩm nằm ở thứ hạng 150-200 có độ liên quan rất thấp nhưng lại sở hữu các thể loại cực kỳ hiếm/khác biệt so với danh sách đang chọn. Phép trừ phạt đa dạng sẽ vô tình đẩy các sản phẩm kém chất lượng này lên Top 10, làm loãng các sản phẩm thực sự phù hợp với người dùng. Do đó, việc giới hạn `rerank_pool_k = 40` vừa là giải pháp tối ưu độ trễ vừa là hàng rào bảo vệ độ chính xác (relevance guardrail).
</details>

---

## 📜 Giấy Phép (License)

Dự án được phát triển và tối ưu hóa bởi **AI Engineer Portfolio**.
Phát hành theo giấy phép [MIT License](LICENSE).
