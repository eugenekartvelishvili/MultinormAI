# config.py  (code/retrieval/rag/config.py)
from code.ingestion import config as ingestion_config

# LLM для классификации и переформулировки
LOCAL_LLM_MODEL = ingestion_config.LLM_MODEL

# LLM для генерации финального ответа
LOCAL_LLM_MODEL_ANSWER = ingestion_config.LLM_MODEL_ANSWER

LLM_BASE_URL = ingestion_config.LLM_BASE_URL

# Milvus
MILVUS_HOST = ingestion_config.MILVUS_HOST
MILVUS_PORT = ingestion_config.MILVUS_PORT
MILVUS_COLLECTION = ingestion_config.COLLECTION_NAME

# ── Пороги score для адаптивного fetch ───────────────────────────────────────

# Точечный fetch: подпункт + заголовок раздела
SCORE_EXACT_SUBSECTION = 0.82

# Fetch подпункта с соседями (±2 по номеру)
SCORE_SUBSECTION_WITH_NEIGHBORS = 0.72

# Минимальный gap между топ-1 и топ-2 документом чтобы брать только один
SCORE_DOC_GAP_THRESHOLD = 0.02

# Максимальный gap между топ-1 и топ-2 разделом чтобы брать оба (стратегия A).
# Если gap < порога — разделы близки по score, берём оба.
# Если gap >= порога — первый явно лучше, берём только его.
SCORE_SECTION_GAP_THRESHOLD = 0.01

# Максимальный gap между топ-1 и следующим разделом чтобы брать оба (стратегия B).
# Стратегия B работает с broad_overview/procedure где контекст шире,
# поэтому порог жёстче — берём дополнительный раздел только если он очень близок.
SCORE_SECTION_GAP_THRESHOLD_B = 0.03

# Порог уверенности pass1 по summary в стратегиях B и C
SCORE_SUMMARY_CONFIDENT = 0.70

# ── Лимиты контекста ─────────────────────────────────────────────────────────

# Максимум символов в итоговом контексте для LLM
CONTEXT_MAX_CHARS = 20000

# Сколько разделов раскрывать по умолчанию
RAG_TOP_SECTIONS = 2

SCORE_ANCHOR_SECTION_BONUS = 0.05