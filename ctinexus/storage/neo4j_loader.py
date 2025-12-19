# import logging
# import re
# import os
# import litellm  # Tận dụng thư viện có sẵn của dự án
# from neo4j import GraphDatabase

# logger = logging.getLogger(__name__)

# class Neo4jLoader:
#     def __init__(self, uri, user, password, database="neo4j", embedding_model=None):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))
#         self.database = database
        
#         # Tận dụng model mặc định của dự án nếu không truyền vào
#         # Mặc định CTINexus dùng 'text-embedding-3-large' hoặc lấy từ ENV
#         self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
#         # Cấu hình cho Ollama/Local nếu cần (giống logic trong graph_constructor.py)
#         self.api_base = os.getenv("OLLAMA_BASE_URL", None)
#         if "llama" in self.embedding_model or "nomic" in self.embedding_model:
#              if not self.embedding_model.startswith("ollama/"):
#                 self.embedding_model = f"ollama/{self.embedding_model}"

#         logger.info(f"🔌 Neo4jLoader sử dụng model embedding: {self.embedding_model}")
        
#         self.verify_connection()
#         self.create_initial_constraints_and_indexes()

#     def close(self):
#         self.driver.close()

#     def verify_connection(self):
#         try:
#             self.driver.verify_connectivity()
#         except Exception as e:
#             logger.error(f"❌ Lỗi kết nối Neo4j: {e}")
#             raise

#     def create_initial_constraints_and_indexes(self):
#         """
#         Tạo Index Vector và Constraints
#         """
#         with self.driver.session(database=self.database) as session:
#             # 1. Constraints (Tính duy nhất)
#             session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
#             session.run("CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (r:Report) REQUIRE r.name IS UNIQUE")
            
#             # 2. Vector Index (Tìm kiếm tương đồng)
#             # Lưu ý: dimensions phải khớp với model. OpenAI large là 3072, Ada-002 là 1536.
#             # Ta để mặc định 1536 (thường dùng) hoặc 3072. Nếu sai Neo4j sẽ báo lỗi khi insert.
#             # Ở đây ta giả định model trả về 1536 hoặc 3072, Neo4j 5.x tự động check.
#             # Tuy nhiên, cần lưu ý: Bạn phải xóa index cũ nếu đổi model có chiều vector khác.
#             try:
#                 session.run("""
#                     CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
#                     FOR (e:Entity) ON (e.embedding)
#                     OPTIONS {indexConfig: {
#                      `vector.dimensions`: 3072, 
#                      `vector.similarity_function`: 'cosine'
#                     }}
#                 """)
#             except Exception as e:
#                 logger.warning(f"⚠️ Không thể tạo Vector Index (có thể do phiên bản Neo4j cũ hoặc đã tồn tại khác config): {e}")

#     def _get_embedding(self, text):
#         """
#         Tận dụng LiteLLM để lấy vector, giống hệt cách Merger của CTINexus làm.
#         """
#         try:
#             # Gọi API qua litellm (hỗ trợ OpenAI, Azure, Ollama...)
#             response = litellm.embedding(
#                 model=self.embedding_model,
#                 input=[text],
#                 api_base=self.api_base
#             )
#             return response["data"][0]["embedding"]
#         except Exception as e:
#             logger.error(f"Lỗi khi tạo embedding cho '{text}': {e}")
#             return None

#     def _normalize_label(self, label):
#         # if not label: return "Entity"
#         return re.sub(r'[^a-zA-Z0-9]', '_', label)

#     def _normalize_relation(self, relation_text):
#         if not relation_text: return "RELATED_TO"
#         return re.sub(r'\s+', '_', relation_text.strip()).upper()

#     def ingest_report(self, cti_result, report_name="Unknown_Report"):
#         ea_triplets = cti_result.get("EA", {}).get("aligned_triplets", [])
#         lp_links = cti_result.get("LP", {}).get("predicted_links", [])
        
#         all_triplets = []
#         for t in ea_triplets:
#             t['is_predicted'] = False
#             all_triplets.append(t)
#         for t in lp_links:
#             t['is_predicted'] = True
#             all_triplets.append(t)

#         with self.driver.session(database=self.database) as session:
#             session.execute_write(self._process_batch, all_triplets, report_name)

#     def _process_batch(self, tx, triplets, report_name):
#         # 1. Vẫn tạo Node Report (để lưu metadata như ngày giờ ingest), nhưng KHÔNG nối cạnh
#         # Node này chỉ dùng để quản lý danh sách các báo cáo đã nạp
#         tx.run("""
#             MERGE (r:Report {name: $report_name})
#             ON CREATE SET r.ingested_at = datetime()
#         """, report_name=report_name)

#         for item in triplets:
#             subj = item.get("subject", {})
#             obj = item.get("object", {})
#             relation = item.get("relation", "RELATED_TO")
#             is_predicted = item.get("is_predicted", False)

#             s_name = subj.get("entity_text", "Unknown")
#             s_type = self._normalize_label(subj.get("mention_class"))
#             o_name = obj.get("entity_text", "Unknown")
#             o_type = self._normalize_label(obj.get("mention_class"))
#             rel_type = self._normalize_relation(relation)

#             # Tạo embedding (giữ nguyên logic cũ)
#             s_emb = self._get_embedding(s_name)
#             o_emb = self._get_embedding(o_name)

#             # --- LOGIC TỐI ƯU ---
#             # Thay vì tạo quan hệ MENTIONED_IN, ta cập nhật thuộc tính 'sources' 
#             # trên mối quan hệ chính.
            
#             cypher_query = f"""
#             // 1. Xử lý Subject (Tìm kiếm vector hoặc merge tên)
#             MERGE (s:Entity {{name: $s_name}})
#             ON CREATE SET s.type = $s_type, s.embedding = $s_emb
#             SET s:{s_type}

#             // 2. Xử lý Object
#             MERGE (o:Entity {{name: $o_name}})
#             ON CREATE SET o.type = $o_type, o.embedding = $o_emb
#             SET o:{o_type}

#             // 3. Xử lý Link & Nguồn gốc (Provenance)
#             MERGE (s)-[r:`{rel_type}`]->(o)
            
#             // Nếu quan hệ mới tạo: Khởi tạo danh sách nguồn
#             ON CREATE SET 
#                 r.is_predicted = $is_predicted, 
#                 r.weight = 1,
#                 r.sources = [$report_name],       // <--- Lưu tên report vào list
#                 r.last_seen = datetime()

#             // Nếu quan hệ đã có: Cập nhật thêm nguồn vào danh sách (nếu chưa có)
#             ON MATCH SET 
#                 r.weight = r.weight + 1,
#                 r.last_seen = datetime(),
#                 r.sources = CASE 
#                     WHEN NOT $report_name IN r.sources THEN r.sources + $report_name 
#                     ELSE r.sources 
#                 END
#             """

#             tx.run(cypher_query, 
#                    s_name=s_name, s_type=s_type, s_emb=s_emb,
#                    o_name=o_name, o_type=o_type, o_emb=o_emb,
#                    is_predicted=is_predicted, report_name=report_name)
import logging
import re
import os
import hashlib
import litellm
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class Neo4jLoader:
    def __init__(self, config):
        # Lấy config
        uri = config.get("neo4j_uri", "bolt://localhost:7687")
        user = config.get("neo4j_user", "neo4j")
        password = config.get("neo4j_password", "password")
        self.database = config.get("neo4j_database", "neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Cấu hình Embedding
        self.embedding_model = config.get("embedding_model", "text-embedding-3-large")
        self.api_base = config.get("ollama_base_url", os.getenv("OLLAMA_BASE_URL"))
        self.api_key = config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

        # Kiểm tra APOC và khởi tạo Index
        self._check_apoc()
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _check_apoc(self):
        """Kiểm tra xem plugin APOC đã được cài đặt chưa"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN apoc.version() AS version")
                version = result.single()
                if version:
                    logger.info(f"✅ Đã tìm thấy APOC version: {version['version']}")
                else:
                    logger.warning("⚠️ Không tìm thấy APOC. Một số tính năng dynamic label có thể bị hạn chế.")
        except Exception as e:
            logger.warning(f"⚠️ Lỗi khi kiểm tra APOC: {e}")

    def _create_constraints(self):
        """Tạo Constraint và Vector Index"""
        with self.driver.session(database=self.database) as session:
            # 1. Đảm bảo tên Entity là duy nhất (Primary Key) -> Giải quyết vấn đề trùng ID
            session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            
            # 2. Đảm bảo Report ID là duy nhất
            session.run("CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (r:Report) REQUIRE r.id IS UNIQUE")

            # 3. Tạo Vector Index (Ví dụ dimension=3072 cho text-embedding-3-large)
            # Lưu ý: Cần điều chỉnh dimensions tùy theo model bạn dùng (OpenAI large=3072, small=1536)
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
                logger.warning(f"⚠️ Vector Index Creation skipped/failed: {e}")

    def _get_embedding(self, text):
        if not text: return None
        try:
            # Xử lý prefix model cho litellm
            model_name = self.embedding_model
            if "llama" in model_name and not model_name.startswith("ollama/"):
                model_name = f"ollama/{model_name}"

            response = litellm.embedding(
                model=model_name,
                input=[text],
                api_base=self.api_base,
                api_key=self.api_key
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed for '{text[:20]}...': {e}")
            return []

    def _normalize_label(self, label):
        """Chuyển 'Indicator:IP Address' -> 'Indicator_IP_Address' để làm Label hợp lệ"""
        if not label or label == "default":
            return "Entity"
        # Chỉ giữ lại chữ cái, số và gạch dưới
        clean = re.sub(r'[^a-zA-Z0-9]', '_', label)
        return clean

    def load_report_data(self, report_json):
        """
        Hàm chính để load dữ liệu. Sử dụng Batch Processing để tối ưu tốc độ.
        """
        # 1. Xử lý thông tin Report
        full_text = report_json.get("text", "")
        # Tạo ID duy nhất cho report dựa trên nội dung (tránh load trùng lặp)
        report_id = hashlib.md5(full_text.encode('utf-8')).hexdigest()
        report_title = full_text[:50].replace("\n", " ") + "..." # Lấy 50 ký tự đầu làm tiêu đề

        # 2. Chuẩn bị dữ liệu Triplets (Cạnh)
        ea_triplets = report_json.get("EA", {}).get("aligned_triplets", [])
        lp_links = report_json.get("LP", {}).get("predicted_links", [])

        # Gộp cả 2 nguồn, đánh dấu loại
        batch_data = []
        
        # Helper function để xử lý từng item
        def prepare_item(item, is_predicted):
            subj = item.get("subject", {})
            obj = item.get("object", {})
            
            s_text = subj.get("entity_text")
            o_text = obj.get("entity_text")
            
            if not s_text or not o_text: return # Bỏ qua nếu thiếu dữ liệu

            return {
                "s_name": s_text,
                "s_type": self._normalize_label(subj.get("mention_class")),
                "s_emb": self._get_embedding(s_text), # Gọi embedding (có thể chậm, nên cache nếu được)
                
                "o_name": o_text,
                "o_type": self._normalize_label(obj.get("mention_class")),
                "o_emb": self._get_embedding(o_text),
                
                "relation": self._normalize_label(item.get("relation", "RELATED_TO")).upper(),
                "mitre_id": item.get("mitre_id"), # Nếu đã chạy qua mapper
                "is_predicted": is_predicted
            }

        logger.info("Processing embeddings and preparing batch...")
        for t in ea_triplets:
            data = prepare_item(t, is_predicted=False)
            if data: batch_data.append(data)
            
        for t in lp_links:
            data = prepare_item(t, is_predicted=True)
            if data: batch_data.append(data)

        if not batch_data:
            logger.warning("No valid triplets found to insert.")
            return

        # 3. Thực thi Cypher (Batch Insert)
        logger.info(f"Inserting {len(batch_data)} elements into Neo4j for report {report_id}...")
        
        query = """
        // 1. Tạo Node Report (Lưu full text nội dung)
        MERGE (r:Report {id: $report_id})
        ON CREATE SET 
            r.content = $full_text,
            r.title = $report_title,
            r.ingested_at = datetime()
        
        // 2. UNWIND: Kỹ thuật xử lý hàng loạt cực nhanh
        WITH r
        UNWIND $batch AS row
        
        // --- Xử lý Subject ---
        // Dùng MERGE theo tên -> Giải quyết việc ID bị reset ở mỗi report
        MERGE (s:Entity {name: row.s_name})
        ON CREATE SET s.embedding = row.s_emb
        // Sử dụng APOC để gán label động (Flexible Node Types)
        // Luôn giữ label :Entity, thêm label class cụ thể (VD: :Malware)
        with r, s, row
        CALL apoc.create.addLabels(s, [row.s_type]) YIELD node as s_node
        
        // --- Xử lý Object ---
        MERGE (o:Entity {name: row.o_name})
        ON CREATE SET o.embedding = row.o_emb
        with r, s, o, row
        CALL apoc.create.addLabels(o, [row.o_type]) YIELD node as o_node
        
        // --- Xử lý Quan hệ & Provenance (Nguồn gốc) ---
        // Sử dụng APOC để tạo quan hệ động (Dynamic Relationship Type)
        // Thay vì fix cứng [:ACTION], ta dùng row.relation (VD: [:TARGETS], [:USES])
        CALL apoc.merge.relationship(s, row.relation, {}, {}, o, {}) YIELD rel
        
        // Cập nhật thuộc tính trên cạnh
        SET rel.mitre_id = row.mitre_id,
            rel.last_updated = datetime()
        
        // --- PROVENANCE MAPPING ---
        // Đây là cách giải quyết vấn đề "quan hệ quá nhiều"
        // Thay vì tạo cạnh (Report)->(Entity), ta lưu Report ID vào một mảng trên cạnh (Relation)
        // Nghĩa là: "Mối quan hệ này được xác nhận bởi các báo cáo nào?"
        SET rel.source_reports = CASE 
            WHEN rel.source_reports IS NULL THEN [$report_id]
            WHEN NOT $report_id IN rel.source_reports THEN rel.source_reports + $report_id
            ELSE rel.source_reports
        END
        
        // (Tùy chọn) Chỉ tạo quan hệ MENTIONS từ Report tới Topic chính (nếu cần)
        // Ở đây ta bỏ qua để đồ thị đỡ rối, vì đã có source_reports trên cạnh.
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, 
                            report_id=report_id, 
                            full_text=full_text, 
                            report_title=report_title,
                            batch=batch_data)
            logger.info("✅ Neo4j Import Successful!")
            
        except Exception as e:
            logger.error(f"❌ Neo4j Import Failed: {e}")