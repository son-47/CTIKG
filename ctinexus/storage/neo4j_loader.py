import logging
import re
import os
import litellm  # Tận dụng thư viện có sẵn của dự án
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class Neo4jLoader:
    def __init__(self, uri, user, password, database="neo4j", embedding_model=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Tận dụng model mặc định của dự án nếu không truyền vào
        # Mặc định CTINexus dùng 'text-embedding-3-large' hoặc lấy từ ENV
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Cấu hình cho Ollama/Local nếu cần (giống logic trong graph_constructor.py)
        self.api_base = os.getenv("OLLAMA_BASE_URL", None)
        if "llama" in self.embedding_model or "nomic" in self.embedding_model:
             if not self.embedding_model.startswith("ollama/"):
                self.embedding_model = f"ollama/{self.embedding_model}"

        logger.info(f"🔌 Neo4jLoader sử dụng model embedding: {self.embedding_model}")
        
        self.verify_connection()
        self.create_initial_constraints_and_indexes()

    def close(self):
        self.driver.close()

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối Neo4j: {e}")
            raise

    def create_initial_constraints_and_indexes(self):
        """
        Tạo Index Vector và Constraints
        """
        with self.driver.session(database=self.database) as session:
            # 1. Constraints (Tính duy nhất)
            session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (r:Report) REQUIRE r.name IS UNIQUE")
            
            # 2. Vector Index (Tìm kiếm tương đồng)
            # Lưu ý: dimensions phải khớp với model. OpenAI large là 3072, Ada-002 là 1536.
            # Ta để mặc định 1536 (thường dùng) hoặc 3072. Nếu sai Neo4j sẽ báo lỗi khi insert.
            # Ở đây ta giả định model trả về 1536 hoặc 3072, Neo4j 5.x tự động check.
            # Tuy nhiên, cần lưu ý: Bạn phải xóa index cũ nếu đổi model có chiều vector khác.
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {indexConfig: {
                     `vector.dimensions`: 3072, 
                     `vector.similarity_function`: 'cosine'
                    }}
                """)
            except Exception as e:
                logger.warning(f"⚠️ Không thể tạo Vector Index (có thể do phiên bản Neo4j cũ hoặc đã tồn tại khác config): {e}")

    def _get_embedding(self, text):
        """
        Tận dụng LiteLLM để lấy vector, giống hệt cách Merger của CTINexus làm.
        """
        try:
            # Gọi API qua litellm (hỗ trợ OpenAI, Azure, Ollama...)
            response = litellm.embedding(
                model=self.embedding_model,
                input=[text],
                api_base=self.api_base
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding cho '{text}': {e}")
            return None

    def _normalize_label(self, label):
        # if not label: return "Entity"
        return re.sub(r'[^a-zA-Z0-9]', '_', label)

    def _normalize_relation(self, relation_text):
        if not relation_text: return "RELATED_TO"
        return re.sub(r'\s+', '_', relation_text.strip()).upper()

    def ingest_report(self, cti_result, report_name="Unknown_Report"):
        ea_triplets = cti_result.get("EA", {}).get("aligned_triplets", [])
        lp_links = cti_result.get("LP", {}).get("predicted_links", [])
        
        all_triplets = []
        for t in ea_triplets:
            t['is_predicted'] = False
            all_triplets.append(t)
        for t in lp_links:
            t['is_predicted'] = True
            all_triplets.append(t)

        with self.driver.session(database=self.database) as session:
            session.execute_write(self._process_batch, all_triplets, report_name)

    def _process_batch(self, tx, triplets, report_name):
        # 1. Vẫn tạo Node Report (để lưu metadata như ngày giờ ingest), nhưng KHÔNG nối cạnh
        # Node này chỉ dùng để quản lý danh sách các báo cáo đã nạp
        tx.run("""
            MERGE (r:Report {name: $report_name})
            ON CREATE SET r.ingested_at = datetime()
        """, report_name=report_name)

        for item in triplets:
            subj = item.get("subject", {})
            obj = item.get("object", {})
            relation = item.get("relation", "RELATED_TO")
            is_predicted = item.get("is_predicted", False)

            s_name = subj.get("entity_text", "Unknown")
            s_type = self._normalize_label(subj.get("mention_class"))
            o_name = obj.get("entity_text", "Unknown")
            o_type = self._normalize_label(obj.get("mention_class"))
            rel_type = self._normalize_relation(relation)

            # Tạo embedding (giữ nguyên logic cũ)
            s_emb = self._get_embedding(s_name)
            o_emb = self._get_embedding(o_name)

            # --- LOGIC TỐI ƯU ---
            # Thay vì tạo quan hệ MENTIONED_IN, ta cập nhật thuộc tính 'sources' 
            # trên mối quan hệ chính.
            
            cypher_query = f"""
            // 1. Xử lý Subject (Tìm kiếm vector hoặc merge tên)
            MERGE (s:Entity {{name: $s_name}})
            ON CREATE SET s.type = $s_type, s.embedding = $s_emb
            SET s:{s_type}

            // 2. Xử lý Object
            MERGE (o:Entity {{name: $o_name}})
            ON CREATE SET o.type = $o_type, o.embedding = $o_emb
            SET o:{o_type}

            // 3. Xử lý Link & Nguồn gốc (Provenance)
            MERGE (s)-[r:`{rel_type}`]->(o)
            
            // Nếu quan hệ mới tạo: Khởi tạo danh sách nguồn
            ON CREATE SET 
                r.is_predicted = $is_predicted, 
                r.weight = 1,
                r.sources = [$report_name],       // <--- Lưu tên report vào list
                r.last_seen = datetime()

            // Nếu quan hệ đã có: Cập nhật thêm nguồn vào danh sách (nếu chưa có)
            ON MATCH SET 
                r.weight = r.weight + 1,
                r.last_seen = datetime(),
                r.sources = CASE 
                    WHEN NOT $report_name IN r.sources THEN r.sources + $report_name 
                    ELSE r.sources 
                END
            """

            tx.run(cypher_query, 
                   s_name=s_name, s_type=s_type, s_emb=s_emb,
                   o_name=o_name, o_type=o_type, o_emb=o_emb,
                   is_predicted=is_predicted, report_name=report_name)