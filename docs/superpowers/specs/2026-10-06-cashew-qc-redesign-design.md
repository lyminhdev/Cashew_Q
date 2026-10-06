# Thiết kế lại Cashew QC và thư viện ảnh mẫu

## Mục tiêu

Thiết kế lại webapp kiểm định hạt điều để người vận hành có thể hoàn thành luồng `chọn ảnh → phân tích → đọc kết quả` mà không cần hiểu các thông số mô hình. Ứng dụng vẫn dùng nguyên pipeline hiện có: YOLO phát hiện hạt, sau đó EfficientNet-B3 hoặc ResNet-50 phân loại sáu lớp.

Giao diện mặc định dùng tông sáng, dễ đọc, tiếng Việt. Chế độ kỹ thuật vẫn cho phép xem confidence và xác suất từng lớp, nhưng các chỉ số này không lấn át kết quả nghiệp vụ.

## Phạm vi

1. Thay thế giao diện dashboard hiện tại bằng trang đơn giản theo ba trạng thái: chưa chọn ảnh, đang phân tích, có kết quả.
2. Thêm thư viện 12 ảnh mẫu: hai ảnh cho mỗi lớp `tb`, `loai1`, `loai2`, `loai3`, `lbw`, và `bad_output`.
3. Các ảnh mẫu được tuyển chọn từ thư mục nguồn hậu tố `_yl`; ảnh phát hành được sao chép thành tài nguyên web tĩnh, không phục vụ toàn bộ dataset.
4. Làm rõ trạng thái API/model và thêm thao tác thử lại khi không kết nối được backend.
5. Dùng Three.js cho minh họa 3D và animation tiến trình nhẹ, có fallback khi WebGL không khả dụng hoặc người dùng bật reduced motion.

Không thay đổi trọng số model, quy tắc phát hiện, nhãn lớp, hoặc hợp đồng `POST /predict` hiện có.

## Người dùng và luồng

### Người vận hành

1. Mở ứng dụng và thấy trạng thái `Sẵn sàng phân tích` cùng model đang hoạt động.
2. Chọn ảnh từ máy hoặc chọn một ảnh trong thư viện mẫu.
3. Bấm nút chính `Phân tích ảnh`.
4. Xem tổng hạt, kết quả chính, tỷ lệ đạt, số hạt cần kiểm tra và ảnh gốc có bounding box.
5. Mở `Xem chi tiết` khi cần đối chiếu từng hạt.

### Kỹ thuật viên

1. Mở khối `Thông tin kỹ thuật` và chọn classifier.
2. Xem độ tin cậy YOLO, độ tin cậy classifier, phân phối xác suất, latency và thiết bị.
3. Dùng cùng dữ liệu kết quả mà người vận hành đang xem; không chạy API riêng hoặc suy luận lại.

## Cấu trúc thông tin

### Header

- Tên sản phẩm `Cashew QC`.
- Badge trạng thái: sẵn sàng, đang tải model, đang phân tích, hoặc không kết nối.
- Badge model: tên classifier và CPU/GPU.
- Chuyển sáng/tối, trong đó sáng là mặc định.

### Trạng thái chưa có ảnh

- Hero ngắn: hướng dẫn một câu, nút `Chọn ảnh` và vùng kéo thả.
- Trình xem Three.js đặt cạnh hero, thể hiện hạt điều 3D chậm và không cạnh tranh với hành động chọn ảnh.
- Thư viện ảnh mẫu theo sáu loại, mỗi thẻ có hai thumbnail và nhãn tiếng Việt.

### Trạng thái đang phân tích

- Giữ ảnh đã chọn tại vị trí ổn định.
- Thể hiện chuỗi tiến trình `Đang gửi ảnh → Đang phát hiện hạt → Đang phân loại → Hoàn tất`.
- Three.js chỉ tạo chuyển động hỗ trợ trong vùng minh họa; không che ảnh hoặc kết quả.

### Trạng thái có kết quả

- Dải tóm tắt: tổng hạt, lớp chiếm đa số, tỷ lệ đạt, số lỗi/cần kiểm tra, thời gian xử lý.
- Ảnh kết quả lớn ở trung tâm với box, nhãn lớp và thao tác chọn từng hạt.
- Khối phân bố chất lượng dễ đọc bằng tên lớp và số lượng, không chỉ dùng màu.
- Bảng chi tiết ẩn sau nút `Xem chi tiết`; khi mở, cho phép lọc theo lớp và trạng thái đạt/lỗi.

## Ảnh mẫu

Nguồn hợp lệ là các thư mục `tb_yl`, `loai1_yl`, `loai2_yl`, `loai3_yl`, `lbw_yl`, và `badoutput_yl`. Việc chọn ảnh không dựa vào tên ảnh đơn lẻ, mà dựa vào manifest được tạo trước khi phát hành.

Mỗi mục manifest có:

- đường dẫn ảnh static;
- `expected_class` theo thư mục nguồn;
- nhãn hiển thị tiếng Việt;
- mã mẫu ổn định;
- kết quả kiểm định model gồm số hạt phát hiện, lớp dự đoán, confidence.

Điều kiện một ảnh được đưa vào thư viện:

1. thuộc đúng thư mục `_yl` của lớp tương ứng;
2. YOLO phát hiện tối thiểu một hạt;
3. dự đoán chính trùng `expected_class` theo ngưỡng tin cậy được ghi rõ trong manifest;
4. khác ảnh cùng loại về khung hình/thời điểm để tránh các ảnh gần như trùng lặp.

Trang chỉ nạp 12 ảnh đã tuyển chọn từ `static/samples/`; ảnh thứ ba cho mỗi loại là lựa chọn mở rộng trong tương lai, không thuộc bản đầu.

## Tích hợp API

Giữ nguyên `GET /health`, `GET /models` và `POST /predict?model=<id>`.

Thêm endpoint/nguồn manifest chỉ nếu cần để frontend đọc danh sách ảnh mẫu; phương án ưu tiên là file JSON tĩnh do frontend tải cùng assets để không tăng rủi ro cho API suy luận. UI chỉ khởi chạy phân tích khi người dùng chủ động bấm nút, kể cả với ảnh mẫu.

## Three.js và khả năng tiếp cận

- Three.js tải theo module riêng và chỉ khởi tạo sau khi vùng hero hiển thị.
- Canvas WebGL có `aria-hidden`; thao tác và kết quả luôn có tương đương bằng HTML.
- `prefers-reduced-motion: reduce` tắt quay, chuyển động camera và hiệu ứng hạt.
- Nếu Three.js/WebGL không tải được, hero tiếp tục hiển thị minh họa tĩnh CSS/SVG.

## Trạng thái lỗi

- Không có kết nối: nêu rõ backend chưa sẵn sàng, có nút `Thử lại`.
- File không phải ảnh hoặc ảnh không đọc được: hiển thị lỗi ngay ở vùng nhập.
- Không có hạt phát hiện: giữ ảnh và giải thích rằng không có hạt đủ điều kiện thay vì hiển thị kết quả rỗng.
- Model không khả dụng: vô hiệu hóa model đó trong bộ chọn và chỉ ra model hiện có.

## Kiểm thử chấp nhận

1. `GET /health` phản ánh đúng trạng thái kết nối trên giao diện.
2. Có 12 mẫu static, đủ hai ảnh cho mỗi sáu lớp; mọi mẫu có metadata đúng.
3. Chọn ảnh mẫu hoặc tải ảnh đều đi vào cùng một luồng preview rồi `POST /predict`.
4. Kết quả từ API vẫn hiển thị box, tổng số, lớp và confidence đúng dữ liệu trả về.
5. Giao diện dùng được ở màn hình desktop và mobile; không cần WebGL vẫn dùng đủ chức năng.
6. Bật reduced motion không có animation lặp liên tục.
