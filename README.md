# blocklist
Filter for AdGuard Home and similar filters to block online gambling and betting websites.


Crawl danh sách các website cờ bạc (seed sites) để thu thập các domain/URL liên quan
và sinh ra bộ lọc dành cho AdGuard Home.
Tính năng:
- Crawl bất đồng bộ (aiohttp) với giới hạn concurrency và rate-limit per-domain
- Lọc link theo từ khóa (casino, bet, poker, betting, sportsbook, slots, gambling...)
- Trích domain bằng tldextract và loại trùng
- Xuất ra 2 định dạng:
    * hosts_like.txt  (0.0.0.0 domain)
    * adguard_filter.txt  (||domain^)

Lưu ý:

"""
- Chạy tool này chỉ để tạo danh sách chặn (blocklist); hãy sử dụng có trách nhiệm.
- Tool cố gắng tôn trọng robots.txt và rate-limit, nhưng crawl quy mô lớn có thể tạo tải.