import logging
import re
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class Neo4jLoader:
    def __init__(self, uri, user, password, database="neo4j"):
        """
        Khởi tạo kết nối đến Neo4j.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.verify_connection()
        self.create_initial_constraints()

    def close(self):
        self.driver.close()

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
            logger.info("✅ Kết nối Neo4j thành công!")
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối Neo4j: {e}")
            raise

    def create_initial_constraints(self):
        """
        Tạo các ràng buộc (Constraints) để đảm bảo tính duy nhất và tối ưu tốc độ tìm kiếm.
        Đây là bước quan trọng để lệnh MERGE hoạt động nhanh.
        """
        queries = [
            # Đảm bảo mỗi Entity là duy nhất dựa trên tên (Name)
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            # Đảm bảo mỗi Report là duy nhất
            "CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (r:Report) REQUIRE r.name IS UNIQUE"
        ]
        
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
        logger.info("Constraints checked/created.")

    def _normalize_label(self, label):
        """
        Chuyển đổi mention_class của CTINexus thành Neo4j Label chuẩn.
        VD: "Indicator: File" -> "Indicator_File"
        VD: "Threat Actor" -> "ThreatActor"
        """
        if not label: 
            return "Entity"
        # Loại bỏ ký tự đặc biệt, thay khoảng trắng/dấu hai chấm bằng _
        clean_label = re.sub(r'[^a-zA-Z0-9]', '_', label)
        return clean_label

    def _normalize_relation(self, relation_text):
        """
        Chuyển đổi quan hệ văn bản thành Relationship Type chuẩn (Uppercase).
        VD: "targeting" -> "TARGETING"
        """
        if not relation_text:
            return "RELATED_TO"
        return re.sub(r'\s+', '_', relation_text.strip()).upper()

    def ingest_report(self, cti_result, report_name="Unknown_Report"):
        """
        Hàm chính để đẩy dữ liệu JSON từ CTINexus vào Neo4j.
        Sử dụng Transaction để đảm bảo toàn vẹn dữ liệu.
        """
        # 1. Lấy dữ liệu Extract (EA) và Predict (LP)
        ea_triplets = cti_result.get("EA", {}).get("aligned_triplets", [])
        lp_links = cti_result.get("LP", {}).get("predicted_links", [])
        
        # Gộp cả 2 nguồn để xử lý một thể
        all_triplets = []
        for t in ea_triplets:
            t['is_predicted'] = False
            all_triplets.append(t)
        for t in lp_links:
            t['is_predicted'] = True
            all_triplets.append(t)

        logger.info(f"Bắt đầu nạp {len(all_triplets)} bộ ba tri thức từ báo cáo: {report_name}")

        with self.driver.session(database=self.database) as session:
            session.execute_write(self._process_batch, all_triplets, report_name)

    def _process_batch(self, tx, triplets, report_name):
        """
        Logic xử lý chi tiết trong một Transaction.
        """
        # 1. Tạo Node cho Báo cáo (để lưu nguồn gốc)
        tx.run("""
            MERGE (r:Report {name: $report_name})
            ON CREATE SET r.ingested_at = datetime()
        """, report_name=report_name)

        for item in triplets:
            subj = item.get("subject", {})
            obj = item.get("object", {})
            relation_text = item.get("relation", "RELATED_TO")
            is_predicted = item.get("is_predicted", False)

            # Chuẩn hóa dữ liệu
            s_name = subj.get("entity_text", subj.get("text", "Unknown"))
            s_type = self._normalize_label(subj.get("mention_class", subj.get("class", "Entity")))
            
            o_name = obj.get("entity_text", obj.get("text", "Unknown"))
            o_type = self._normalize_label(obj.get("mention_class", obj.get("class", "Entity")))
            
            rel_type = self._normalize_relation(relation_text)

            # 2. Câu lệnh Cypher "Thần thánh" (Dùng APOC hoặc logic thủ công để set Label động)
            # Vì Cypher không cho phép tham số hóa Label (vd: :$Label), ta dùng apoc.merge.node 
            # hoặc xử lý logic đơn giản với Label chính là :Entity và property là type.
            # Ở đây tôi dùng cách tối ưu nhất mà không cần cài APOC plugin:
            
            query = f"""
            // --- Xử lý Subject ---
            MERGE (s:Entity {{name: $s_name}})
            ON CREATE SET s.type = $s_type, s.created_at = datetime()
            // Hack để add label động (nếu cần thiết kế phức tạp hơn thì dùng APOC)
            SET s:{s_type}

            // --- Xử lý Object ---
            MERGE (o:Entity {{name: $o_name}})
            ON CREATE SET o.type = $o_type, o.created_at = datetime()
            SET o:{o_type}

            // --- Xử lý Quan hệ ---
            // Merge quan hệ để tránh tạo trùng lặp nếu chạy lại report
            MERGE (s)-[r:`{rel_type}`]->(o)
            ON CREATE SET r.is_predicted = $is_predicted, r.weight = 1
            ON MATCH SET r.weight = r.weight + 1

            // --- Xử lý Nguồn gốc (Provenance) ---
            // Kết nối các Entity này với Report để biết thông tin đến từ đâu
            WITH s, o, r
            MATCH (rpt:Report {{name: $report_name}})
            MERGE (s)-[:MENTIONED_IN]->(rpt)
            MERGE (o)-[:MENTIONED_IN]->(rpt)
            """

            tx.run(query, 
                   s_name=s_name, s_type=s_type,
                   o_name=o_name, o_type=o_type,
                   is_predicted=is_predicted,
                   report_name=report_name)