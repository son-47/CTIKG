import os
import json
from pathlib import Path

def check_mitre_cache():
    """Kiểm tra trạng thái cache MITRE"""
    cache_dir = Path("mitre-ttp-mapping-main")
    
    if not cache_dir.exists():
        print("❌ Thư mục mitre-ttp-mapping-main không tồn tại!")
        return
    
    # Tìm tất cả file cache
    cache_files = list(cache_dir.glob("mitre_cache*.json"))
    
    if not cache_files:
        print("⚠️ Chưa có cache nào. Chạy CTINexus lần đầu sẽ tạo cache.")
        return
    
    for cache_file in cache_files:
        print(f"\n📦 Cache: {cache_file.name}")
        print(f"   Size: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            print(f"   ✅ Valid cache with {len(data)} vectors")
            
            # Kiểm tra cấu trúc
            if data:
                sample = data[0]
                print(f"   Fields: {list(sample.keys())}")
                print(f"   Sample ID: {sample.get('id')}")
                print(f"   Vector dim: {len(sample.get('vec', []))}")
                
        except Exception as e:
            print(f"   ❌ Corrupted cache: {e}")
            print(f"   → Xóa file này: rm {cache_file}")

if __name__ == "__main__":
    check_mitre_cache()