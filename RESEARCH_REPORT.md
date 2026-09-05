# Báo cáo cải tiến dựa trên nghiên cứu

## Nghiên cứu nền tảng

- Covington, Adams & Sargin (RecSys 2016), *Deep Neural Networks for YouTube Recommendations*: https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- Harper & Konstan (2015), *The MovieLens Datasets*: https://doi.org/10.1145/2827872

## Kiến trúc Pipeline qua các phiên bản

- **Trước (v1)**: leave-last-two theo user → SVD candidate score → top 200 → rerank latent + popularity → Recall/NDCG.
- **Phiên bản v2**: time split giữ đúng thứ tự → candidate generation → loại toàn bộ item đã thấy trước khi lấy candidate → rerank → top-k; cold-start dùng popularity. Danh sách `seen` được lưu cùng artifact model. Trọng số latent/popularity được chọn trên 1.000 user validation.
- **Phiên bản chuẩn hóa v3 (Canonical 6-Stage Pipeline)**:
  1. **Data Preparation**: Định nghĩa implicit positive feedback (Rating $\ge$ 4.0), sắp xếp chuỗi thời gian của từng user.
  2. **Temporal Split (Leave-Last-Two Positive)**: Phân định rạch ròi Training Signal (Rating $\ge$ 4.0) vs Seen Filter (mọi item đã xem trong train). Tuyệt đối không để tương tác tương lai lọt vào tập train.
  3. **Candidate Retrieval**: Tách trừu tượng hóa `CandidateRetriever` (`SVDRetriever`, `PopularityRetriever`), áp dụng TruncatedSVD latent factor decomposition. Định vị rõ exact dot product $O(N \cdot d)$ và lộ trình ANN (FAISS/ScaNN).
  4. **Reranking & Diversity**: Tách module hóa `CandidateFeatureBuilder` (Latent + Log1p Popularity Prior + User Genre Affinity) $\rightarrow$ `TwoStageRanker` $\rightarrow$ `DiversityReranker` (MMR-style genre diversity penalty). Tuning đồng thời $\alpha = 0.95$ và $\lambda_{div} = 0.10$ trên validation grid search đa mục tiêu.
  5. **Offline Evaluation**: Đánh giá chính thức trên **toàn bộ 6.035 người dùng hợp lệ** trong tập Test. Bổ sung các chỉ số: HitRate@10 (đồng nhất với Leave-one-out Recall@10), NDCG@10, MRR@10, ILD, Novelty@10, User Coverage, phân bổ phơi nhiễm Long-Tail (Head/Mid/Tail share). Đo lường độ trễ chi tiết từng tầng (Retrieval, Ranking, MMR, Total).
  6. **Online Serving**: FastAPI tích hợp `ColdStartPolicy` thống nhất (Global vs Genre-Aware), trường `strategy` phục vụ giám sát, debug scores và giải thích đề xuất nhẹ.

## Kết quả thực nghiệm v3 trên toàn bộ 6.035 người dùng Test

- **Recall@10 / HitRate@10**: **0.0896** (vượt baseline phổ biến 0.0456, đạt độ nhấc **+96.7%**).
- **NDCG@10**: **0.0445** (vượt baseline phổ biến 0.0219, đạt độ nhấc **+102.8%**).
- **MRR@10**: **0.0309** (vượt baseline phổ biến 0.0150, đạt độ nhấc **+106.7%**).
- **Catalog Coverage**: **23.01%** (gấp 5.7 lần baseline 4.05%).
- **User Coverage**: **100.0%**.
- **Intra-List Diversity (ILD)**: **0.7882**.
- **Novelty@10**: **9.2885** (so với baseline 8.1332).
- **Popular-Item Share**: **50.53%** (cân bằng hài hòa giữa độ hot và tính ngách).
- **Phân bổ phơi nhiễm Long-Tail**: Head: 89.81%, Mid: 10.19%, Tail: 0.00%.

## Báo cáo Ablation Study & Phân rã độ trễ từng tầng (2.000 users)

- **Popularity Baseline**: p50 = 0.28 ms, p95 = 0.72 ms.
- **SVD only (Latent Dot Product)**: Recall@10 = 0.0835, ILD = 0.7572, Coverage = 20.69% | p50 = 1.91 ms (Retrieval: 0.67 ms, Ranking: 1.20 ms, MMR: 0.02 ms).
- **SVD + Popularity Prior**: Recall@10 = 0.0835, ILD = 0.7585, Coverage = 20.42% | p50 = 2.52 ms (Retrieval: 0.82 ms, Ranking: 1.57 ms, MMR: 0.03 ms).
- **SVD + Popularity + MMR (Full Pipeline)**: Recall@10 = 0.0815, ILD = 0.7863, Coverage = 20.31% | p50 = 17.46 ms, p95 = 28.91 ms (Retrieval: 0.79 ms, Ranking: 1.29 ms, MMR: 15.07 ms).

**Nhận định kỹ thuật**:
- MMR Diversity Reranking là nhân tố quyết định nâng cao chỉ số đa dạng ILD (+3.84% gain) với mức suy hao độ chính xác không đáng kể.
- MMR Reranking là điểm nghẽn chính về mặt độ trễ (chiếm ~86% thời gian xử lý), xác định hướng ưu tiên tối ưu hóa bằng bitmask vectorization hoặc ANN trong tương lai.
