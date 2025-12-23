# Cyber Threat Intelligence với Large Language Models

Dự án này hỗ trợ trích xuất thông tin tình báo an ninh mạng (Cyber Threat Intelligence – CTI) từ văn bản tự do và xây dựng đồ thị tri thức (knowledge graph) dựa trên các thực thể và quan hệ được phát hiện.

## Tính năng

- Trích xuất thực thể bảo mật: phần mềm độc hại, lỗ hổng, hạ tầng, chỉ báo tấn công (IOC), tổ chức, v.v.
- Trích xuất quan hệ giữa các thực thể (ai dùng gì, tấn công khi nào, bằng cách nào…)
- Xây dựng và lưu trữ đồ thị tri thức để phân tích và trực quan hóa.
- Hỗ trợ nhiều nhà cung cấp mô hình ngôn ngữ lớn (LLM) khác nhau.
- Có thể chạy qua:
  - Thư viện Python
  - Dòng lệnh (CLI)
  - Giao diện web cục bộ
  - Docker

## Yêu cầu

- Python ≥ 3.10
- Tài khoản/API key cho ít nhất một nhà cung cấp LLM (ví dụ: OpenAI, Google Gemini, AWS Bedrock, Ollama cục bộ)
- Hệ điều hành: Linux, macOS hoặc Windows

## Cài đặt

### Cài từ mã nguồn (khuyên dùng khi phát triển)

```bash
// filepath: d:\ctinexus\README.md
pip install -e .
```

Hoặc cài các phụ thuộc trực tiếp:

```bash
// filepath: d:\ctinexus\README.md
pip install -r requirements.txt
```

### Chạy bằng Docker

```bash
// filepath: d:\ctinexus\README.md
docker compose up --build
# hoặc
docker compose up -d --build
```

Sau đó mở trình duyệt tại: `http://localhost:8000`.

## Cấu hình

Tạo file `.env` ở thư mục gốc dự án (xem mẫu trong `.env.example`). Ví dụ:

```bash
// filepath: d:\ctinexus\README.md
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
OLLAMA_BASE_URL=http://localhost:11434
```

Bạn chỉ cần cấu hình những provider thực sự sử dụng.

## Sử dụng

### 1. Sử dụng như thư viện Python

```python
// filepath: d:\ctinexus\README.md
from dotenv import load_dotenv
from your_package import process_cti_report  # thay your_package bằng tên package của bạn

load_dotenv()

text = """
APT29 used PowerShell to download additional malware from a command-and-control
server. The attack exploited CVE-2023-1234 in Microsoft Exchange.
"""

result = process_cti_report(
    text=text,
    provider="openai",
    model="gpt-4o",
    similarity_threshold=0.6,
    output="results.json",  # tùy chọn: lưu kết quả ra file
)

print(result["entity_relation_graph"])
```

Các tham số phổ biến:

- `text`: nội dung báo cáo / log / phân tích cần xử lý.
- `provider`: nhà cung cấp LLM (`"openai"`, `"gemini"`, `"aws"`, `"ollama"`, …).
- `model`: tên mô hình cụ thể (tùy provider).
- `similarity_threshold`: ngưỡng lọc quan hệ trùng lặp / kém liên quan.
- `output`: đường dẫn file JSON để lưu kết quả (tùy chọn).

### 2. Giao diện web cục bộ

Sau khi cài đặt:

```bash
// filepath: d:\ctinexus\README.md
python -m your_package.app  # thay your_package bằng tên package của bạn
```

Sau đó mở `http://localhost:8000` trong trình duyệt.

Chức năng chính:

1. Dán văn bản CTI vào ô nhập.
2. Chọn provider và model.
3. Bấm nút chạy để phân tích.
4. Xem danh sách thực thể, quan hệ và đồ thị tương tác.
5. Xuất kết quả ra JSON hoặc lưu ảnh đồ thị.

### 3. Dòng lệnh (CLI)

Nếu bạn đã cấu hình entrypoint CLI cho dự án (ví dụ: `your-cli`), có thể:

```bash
// filepath: d:\ctinexus\README.md
# Xử lý file
your-cli --input-file report.txt

# Xử lý trực tiếp từ chuỗi văn bản
your-cli --text "APT29 exploited a vulnerability using PowerShell..."

# Chỉ định provider và model
your-cli -i report.txt --provider openai --model gpt-4o

# Lưu kết quả vào thư mục tùy chọn
your-cli -i report.txt --output results/analysis.json
```

Hãy thay `your-cli` bằng tên lệnh thực tế bạn cấu hình trong `pyproject.toml`.

## Phát triển

### Thiết lập môi trường

```bash
// filepath: d:\ctinexus\README.md
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Chạy test

```bash
// filepath: d:\ctinexus\README.md
pytest tests/ -v
```

### Kiểm tra định dạng & lint

```bash
// filepath: d:\ctinexus\README.md
pre-commit run --all-files
```

## Giấy phép

Mã nguồn được phân phối theo giấy phép MIT (xem chi tiết trong file `LICENSE.txt`).
