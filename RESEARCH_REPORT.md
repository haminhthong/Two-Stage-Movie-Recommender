# Báo cáo cải tiến dựa trên nghiên cứu

## Nghiên cứu nền tảng

- Covington, Adams & Sargin (RecSys 2016), *Deep Neural Networks for YouTube Recommendations*: https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- Harper & Konstan (2015), *The MovieLens Datasets*: https://doi.org/10.1145/2827872

## Pipeline trước và sau

Trước: leave-last-two theo user → SVD candidate score → top 200 → rerank latent + popularity → Recall/NDCG.

Sau: time split giữ đúng thứ tự → candidate generation → loại toàn bộ item đã thấy trước khi lấy candidate → rerank → top-k; cold-start vẫn dùng popularity. Danh sách `seen` được lưu cùng artifact model. Sửa này quan trọng vì đề xuất item đã tương tác làm lãng phí vị trí và khiến offline evaluation/serving không đúng mục tiêu khám phá item mới.

Kiến trúc vẫn theo funnel hai tầng của Covington et al.: tầng candidate tối ưu recall trên corpus lớn, tầng ranking phân biệt chi tiết trong tập nhỏ.

## Đánh giá

- Recall@K, NDCG@K, candidate recall và catalog coverage.
- Cold-start: báo cáo riêng user chưa biết; kiểm tra không có item đã xem trong output.
- Vận hành: p95 latency, tỷ lệ fallback và diversity/novelty.

MovieLens rating là explicit feedback; nhị phân hóa rating chỉ là baseline. Bước tiếp theo: so sánh implicit ALS/BPR, negative sampling, tuning trọng số popularity trên validation, thêm diversity reranking và A/B test vì offline metric không thay thế được hành vi online.

## Kết quả chạy thực tế

Đã train trên MovieLens 1M với ma trận 6.040 × 3.704 và 64 latent dimensions. Trên 2.000 người dùng test: **Recall@10 = 0,0780**, **NDCG@10 = 0,0395**. Metric JSON được lưu trong `reports/test_metrics.json`; output serving đã loại item từng xem.

## Nâng cấp hoàn thiện v2

Trọng số latent/popularity được chọn trên 1.000 user validation, không dùng test. Tầng cuối dùng MMR-style genre penalty có thể điều chỉnh qua API. Trên 2.000 user test: Recall@10 **0,0800**, NDCG@10 **0,03985**, catalog coverage **0,1968**, intra-list diversity **0,7812**, popular-item share **0,1372**. Recall tăng nhẹ so với v1 và hệ thống nay đo được trade-off relevance–diversity thay vì chỉ tối ưu hit rate.
