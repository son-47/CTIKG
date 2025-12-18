import os
import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from ctinexus.utils.model_utils import get_embedding, get_response

class MitreMapper:
    def __init__(self, config, dataset_dir="mitre-ttp-mapping-main"):
        self.config = config
        # Lấy cấu hình model từ file config.yaml
        self.emb_model = config.get("embedding_model", "text-embedding-3-small")
        self.llm_model = config.get("llm_model", "gpt-3.5-turbo")
        self.api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        
        self.vector_db = []
        self.dataset_dir = dataset_dir
        self.cache_file = os.path.join(dataset_dir, "mitre_cache.json")
        
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        # 1. Thử load từ cache cho nhanh
        if os.path.exists(self.cache_file):
            print(f"📦 Loading MITRE knowledge from cache: {self.cache_file}")
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert list back to numpy array for speed
                self.vector_db = [{'vector': np.array(d['vec']), 'id': d['id'], 'text': d['text']} for d in data]
            return

        print("⚠️ Creating MITRE Knowledge Base (First time run)...")
        # 2. Nếu chưa có cache thì tạo mới từ CSV
        # Load Descriptions
        self._load_csv(os.path.join(self.dataset_dir, "datasets/ttp-desc-mappings.csv"), 
                       col_id='technique_id', col_text='description', sep='\t')
        
        # Load Procedures (Quan trọng!)
        self._load_csv(os.path.join(self.dataset_dir, "datasets/procedures/procedure_train.tsv"), 
                       col_id='label', col_text='text', sep='\t')

        # 3. Lưu cache
        if self.vector_db:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump([{'vec': d['vector'].tolist(), 'id': d['id'], 'text': d['text']} for d in self.vector_db], f)
            print("✅ MITRE Cache saved.")

    def _load_csv(self, path, col_id, col_text, sep=','):
        if not os.path.exists(path):
            print(f"❌ Cannot find dataset: {path}")
            return
        try:
            df = pd.read_csv(path, sep=sep, on_bad_lines='skip')
            # Lấy mẫu nhỏ để test nếu máy yếu, bỏ dòng dưới khi chạy thật
            # df = df.head(100) 
            
            print(f"   Embedding {len(df)} rows from {os.path.basename(path)}...")
            for _, row in df.iterrows():
                text = str(row.get(col_text, ''))
                tid = str(row.get(col_id, ''))
                
                if len(text) > 5 and tid.startswith('T'):
                    vec = get_embedding(text, self.emb_model, self.api_key)
                    if vec:
                        self.vector_db.append({
                            "vector": np.array(vec),
                            "id": tid,
                            "text": text[:300] # Cắt ngắn để tiết kiệm token
                        })
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def map_action(self, subject, action, obj, context=""):
        """
        Hàm chính để map hành vi sang MITRE ID
        """
        if not action: return None
        
        # Tạo câu query
        query = f"{subject} {action} {obj}"
        query_vec = get_embedding(query, self.emb_model, self.api_key)
        
        if query_vec is None: return None

        # Giai đoạn 1: Retrieve (Lấy top 5 vector giống nhất)
        candidates = []
        for item in self.vector_db:
            # Cosine similarity
            score = 1 - cosine(query_vec, item['vector'])
            if score > 0.72: # Ngưỡng lọc
                candidates.append((score, item))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:5]
        
        if not top_candidates: return None

        # Giai đoạn 2: Rerank (Dùng LLM kiểm tra)
        return self._llm_verify(query, context, top_candidates)

    def _llm_verify(self, query, context, candidates):
        options = "\n".join([f"- Option {i}: ID={c[1]['id']} | Desc: {c[1]['text']}..." for i, c in enumerate(candidates)])
        
        prompt = f"""
        Act as a Cyber Security Analyst. Map the extracted action to a MITRE ATT&CK technique.
        
        Action extracted: "{query}"
        Context from report: "{context[:400]}..."
        
        Candidates found via vector search:
        {options}
        
        Task: Return the ID of the best matching Option. If none fit well, return "None".
        Response format (JSON): {{"mitre_id": "Txxxx", "confidence": "High/Medium/Low"}}
        """
        
        resp = get_response(prompt, self.llm_model, self.api_key)
        try:
            # Lọc JSON từ phản hồi
            import re
            json_match = re.search(r'\{.*\}', resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data.get('mitre_id') and data['mitre_id'] != "None":
                    return data
        except:
            pass
        return None