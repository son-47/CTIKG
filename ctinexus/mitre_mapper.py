import os
import re
import json
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from scipy.spatial.distance import cosine
from ctinexus.utils.model_utils import get_embedding, get_response

class MitreMapper:
    def __init__(self, config, dataset_dir="mitre-ttp-mapping-main"):
        self.config = config
        self.emb_model = config.get("embedding_model", "text-embedding-3-small")
        self.llm_model = config.get("llm_model", "gpt-3.5-turbo")
        self.api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

        self.vector_db = []
        self.dataset_dir = dataset_dir
        cache_name = f"mitre_cache_{self.emb_model.replace('/', '_')}.json"
        self.cache_file = os.path.join(dataset_dir, cache_name)

        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Load MITRE knowledge base with safe caching (only ttp-desc-mappings.csv)."""
        # 1. Thử load cache
        if os.path.exists(self.cache_file):
            try:
                logger.info(f"📦 Loading MITRE cache: {self.cache_file}")
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not data or not isinstance(data, list):
                    raise ValueError("Invalid cache format")

                required_fields = {"vec", "id", "text"}
                if not all(required_fields.issubset(d.keys()) for d in data[: min(5, len(data))]):
                    raise ValueError("Missing required fields in cache")

                self.vector_db = [
                    {"vector": np.array(d["vec"]), "id": d["id"], "text": d["text"]} for d in data
                ]
                logger.info(f"✅ Loaded {len(self.vector_db)} MITRE vectors from cache")
                return
            except Exception as e:
                logger.warning(f"⚠️ Cache corrupted ({e}), rebuilding...")
                try:
                    os.remove(self.cache_file)
                except Exception:
                    pass

        # 2. Build from scratch (CHỈ embed ttp-desc-mappings.csv)
        logger.info("⚙️ Building MITRE Knowledge Base from ttp-desc-mappings.csv (first run)...")

        # Dataset header thường là: "ttp" \t "description"
        desc_path = os.path.join(self.dataset_dir, "datasets/ttp-desc-mappings.csv")
        self._load_csv(
            desc_path,
            col_id="ttp",        # hoặc 'technique_id' nếu file của bạn dùng header này
            col_text="description",
            sep="\t",            # đổi thành ',' nếu file là CSV chuẩn
            id_is_list=False,
        )


        # 3. Save cache
        if self.vector_db:
            try:
                cache_data = [
                    {"vec": d["vector"].tolist(), "id": d["id"], "text": d["text"]}
                    for d in self.vector_db
                ]
                temp_file = self.cache_file + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2)
                os.replace(temp_file, self.cache_file)
                logger.info(
                    f"✅ MITRE cache saved: {self.cache_file} ({len(cache_data)} vectors)"
                )
            except Exception as e:
                logger.error(f"❌ Failed to save cache: {e}")

    def _load_csv(self, path, col_id, col_text, sep=',', id_is_list=False):
        """Load and embed CSV/TSV data with progress tracking.

        Args:
            path: File path.
            col_id: Column containing MITRE IDs (or list of IDs when id_is_list=True).
            col_text: Column containing free-text description/procedure.
            sep: Column separator.
            id_is_list: When True, "col_id" holds a Python-list-like string and
                        the first element will be used as the technique ID.
        """
        if not os.path.exists(path):
            logger.error(f"❌ Dataset not found: {path}")
            return
        
        try:
            df = pd.read_csv(path, sep=sep, on_bad_lines='skip')
            total = len(df)
            logger.info(f"   Processing {total} rows from {os.path.basename(path)}...")
            
            embedded_count = 0
            for idx, row in df.iterrows():
                text = str(row.get(col_text, '')).strip()
                raw_tid = row.get(col_id, '')

                # Parse MITRE technique ID
                tid = ''
                if id_is_list:
                    # labels column is a list-like string, e.g. "['T1001', 'T1132.001']"
                    try:
                        import ast
                        parsed = ast.literal_eval(str(raw_tid))
                        if isinstance(parsed, (list, tuple)) and parsed:
                            tid = str(parsed[0]).strip()
                    except Exception:
                        tid = ''
                else:
                    tid = str(raw_tid).strip().strip('"')

                # Normalise ID to upper-case, dataset often uses lowercase like "t1055.011"
                tid = tid.upper()
                
                if len(text) > 5 and tid.startswith('T'):
                    vec = get_embedding(text, self.emb_model, self.api_key)
                    if vec is not None:
                        self.vector_db.append({
                            "vector": np.array(vec),
                            "id": tid,
                            "text": text[:200]
                        })
                        embedded_count += 1
                
                # ✅ Progress indicator
                if (idx + 1) % 50 == 0:
                    logger.info(f"      [{idx+1}/{total}] embedded (✅ {embedded_count} successful)...")
            
            logger.info(f"   ✅ Successfully embedded {embedded_count}/{total} entries")
            
        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")

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
            # Fix 1: Thử parse trực tiếp nếu response là pure JSON
            try:
                data = json.loads(resp.strip())
                if data.get('mitre_id') and data['mitre_id'] != "None":
                    return data
            except json.JSONDecodeError:
                pass
            
            # Fix 2: Tìm JSON block trong markdown code fence
            match = re.search(r'```json\s*(\{[^}]+\})\s*```', resp, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if data.get('mitre_id') and data['mitre_id'] != "None":
                    return data
            
            # Fix 3: Tìm JSON object độc lập (non-greedy)
            match = re.search(r'\{[^{}]*"mitre_id"[^{}]*\}', resp)
            if match:
                data = json.loads(match.group())
                if data.get('mitre_id') and data['mitre_id'] != "None":
                    return data
                    
        except Exception as e:
            logger.error(f"Failed to parse MITRE verification response: {e}")
        
        return None