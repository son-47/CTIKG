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
        uri = config.get("NEO4J_URI", "bolt://localhost:7687")
        user = config.get("NEO4J_USER", "neo4j")
        password = config.get("NEO4J_PASSWORD", "password")
        self.database = config.get("NEO4J_DATABASE", "neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Cấu hình Embedding
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
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
            session.run("CREATE CONSTRAINT mitre_id_unique IF NOT EXISTS FOR (m:MitreTechnique) REQUIRE m.id IS UNIQUE")
            # 3. Tạo Vector Index (Ví dụ dimension=3072 cho text-embedding-3-large)
            # Lưu ý: Cần điều chỉnh dimensions tùy theo model bạn dùng (OpenAI large=3072, small=1536)
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {indexConfig: {
                     `vector.dimensions`: 1536,
                     `vector.similarity_function`: 'cosine'
                    }}
                """)
            except Exception as e:
                logger.warning(f"⚠️ Vector Index Creation skipped/failed: {e}")


    def _normalize_label(self, label):
        """Chuyển 'Indicator:IP Address' -> 'Indicator_IP_Address' để làm Label hợp lệ"""
        if not label or label == "default":
            return "Entity"
        # Chỉ giữ lại chữ cái, số và gạch dưới
        clean = re.sub(r'[^a-zA-Z0-9]', '_', label)
        return clean

    def _get_embedding(self, text):
        """Trả về list[float] để Neo4j lưu dạng array; trả None nếu lỗi."""
        if not text or not self.embedding_model:
            return None
        try:
            model_name = self.embedding_model
            if "llama" in model_name and not model_name.startswith("ollama/"):
                model_name = f"ollama/{model_name}"

            resp = litellm.embedding(
                model=model_name,
                input=[text],
                api_base=self.api_base,
                api_key=self.api_key,
            )
            emb = resp["data"][0]["embedding"]
            return [float(x) for x in emb]
        except Exception as e:
            logger.error(f"Embedding failed for '{text[:30]}...': {e}")
            return None

    def load_report_data(self, report_json):
        """Đẩy cả EA + LP + Topic vào Neo4j, kèm MITRE mapping."""
        full_text = report_json.get("text", "")
        report_id = hashlib.md5(full_text.encode("utf-8")).hexdigest()

        ea_triplets = report_json.get("EA", {}).get("aligned_triplets", [])
        lp_data = report_json.get("LP", {}) or {}
        lp_links = lp_data.get("predicted_links", [])
        topic_node = lp_data.get("topic_node") or {}

        def get_clean_id(text: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]", "_", text.strip().lower())

        def norm_label(label: str) -> str:
            if not label or label == "default":
                return "Entity"
            return re.sub(r"[^a-zA-Z0-9]", "_", str(label))

        def clean_mitre_id(v):
            if not v:
                return None
            v_str = str(v).strip()
            if not v_str or v_str.upper() == "NONE":
                return None
            return v_str.upper()

        def to_confidence(v):
            """mitre_mapper trả High/Medium/Low; normalize về float."""
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip().lower()
            if s in ("high", "h"):
                return 0.9
            if s in ("medium", "med", "m"):
                return 0.6
            if s in ("low", "l"):
                return 0.3
            try:
                return float(s)
            except Exception:
                return None

        # Build index từ EA theo entity_id để enrich LP nodes (LP.object thường thiếu class/text)
        entity_index = {}
        for t in ea_triplets:
            for side in ("subject", "object"):
                n = (t.get(side) or {})
                eid = n.get("entity_id")
                if eid is not None and eid not in entity_index:
                    entity_index[eid] = n

        def resolve_node(raw: dict):
            raw = raw or {}
            eid = raw.get("entity_id")
            ref = entity_index.get(eid) if eid is not None else None

            # Ưu tiên entity_text từ EA index để đảm bảo cùng ID/label
            text = None
            if ref:
                text = ref.get("entity_text") or ref.get("mention_text")
            text = text or raw.get("entity_text") or raw.get("mention_text")

            cls = raw.get("mention_class") or (ref.get("mention_class") if ref else None)
            cls = norm_label(cls)

            return text, cls

        batch = []

        def prepare_triplet(item, is_predicted: bool = False):
            subj = item.get("subject", {}) or {}
            obj = item.get("object", {}) or {}

            s_text, s_type = resolve_node(subj)
            o_text, o_type = resolve_node(obj)
            if not s_text or not o_text:
                return None

            mitre_id = clean_mitre_id(item.get("mitre_id"))
            mitre_conf = to_confidence(item.get("mitre_confidence"))

            return {
                "s_id": get_clean_id(s_text),
                "s_name": s_text.strip(),
                "s_type": s_type,

                "o_id": get_clean_id(o_text),
                "o_name": o_text.strip(),
                "o_type": o_type,

                "relation": norm_label(item.get("relation", "RELATED_TO")).upper(),
                "mitre_id": mitre_id,
                "mitre_conf": mitre_conf,
                "is_predicted": bool(is_predicted),
            }

        for t in ea_triplets:
            d = prepare_triplet(t, is_predicted=False)
            if d:
                batch.append(d)

        for t in lp_links:
            d = prepare_triplet(t, is_predicted=True)
            if d:
                batch.append(d)

        # Topic node (Report -> Topic) (không set embedding)
        topic = None
        if topic_node:
            t_text = topic_node.get("entity_text") or topic_node.get("mention_text")
            t_cls = topic_node.get("mention_class")
            if t_text:
                topic = {
                    "id": get_clean_id(t_text),
                    "name": t_text.strip(),
                    "type": norm_label(t_cls),
                }

        if not batch and not topic:
            logger.warning("⚠️ No data found to insert.")
            return

        query_main = """
        MERGE (r:Report {id: $report_id})
        ON CREATE SET
            r.content = $full_text,
            r.ingested_at = datetime()

        WITH r
        UNWIND $batch AS row

        // MERGE theo id KHÔNG kèm label => tránh duplicate toàn cục
        MERGE (s {id: row.s_id})
        ON CREATE SET s.name = row.s_name
        ON MATCH  SET s.name = coalesce(s.name, row.s_name)
        WITH r, row, s
        CALL apoc.create.setLabels(s, [row.s_type]) YIELD node AS s2

        MERGE (o {id: row.o_id})
        ON CREATE SET o.name = row.o_name
        ON MATCH  SET o.name = coalesce(o.name, row.o_name)
        WITH r, row, s2, o
        CALL apoc.create.setLabels(o, [row.o_type]) YIELD node AS o2

        CALL apoc.merge.relationship(s2, row.relation, {}, {}, o2, {}) YIELD rel
        SET rel.mitre_id     = row.mitre_id,
            rel.last_updated = datetime(),
            rel.is_predicted = row.is_predicted,
            rel.source_reports =
                CASE
                    WHEN rel.source_reports IS NULL THEN [$report_id]
                    WHEN NOT $report_id IN rel.source_reports THEN rel.source_reports + $report_id
                    ELSE rel.source_reports
                END

        FOREACH (_ IN CASE WHEN row.mitre_id IS NULL THEN [] ELSE [1] END |
            MERGE (mt:MitreTechnique {id: row.mitre_id})
            ON CREATE SET mt.url = 'https://attack.mitre.org/techniques/' + row.mitre_id
            MERGE (s2)-[mr:USES_TECHNIQUE]->(mt)
            SET mr.confidence = coalesce(row.mitre_conf, 0.5),
                mr.from_report = $report_id
        )
        """

        query_topic = """
        MATCH (r:Report {id: $report_id})
        WITH r, $topic AS topic
        WHERE topic IS NOT NULL
        MERGE (t {id: topic.id})
        ON CREATE SET t.name = topic.name
        ON MATCH  SET t.name = coalesce(t.name, topic.name)
        WITH r, t, topic
        CALL apoc.create.setLabels(t, [topic.type]) YIELD node AS t2
        MERGE (r)-[:DISCUSSES_TOPIC]->(t2)
        """

        try:
            with self.driver.session(database=self.database) as session:
                session.run(query_main, report_id=report_id, full_text=full_text, batch=batch)
                if topic:
                    session.run(query_topic, report_id=report_id, topic=topic)
            logger.info("✅ Neo4j Import Successful!")
        except Exception as e:
            logger.error(f"❌ Neo4j Import Failed: {e}")
            raise
            

   
   