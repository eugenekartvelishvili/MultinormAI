import logging
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

from code.ingestion import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _build_schema() -> CollectionSchema:
    """
    Схема коллекции documents.
    Все дополнительные поля — dynamic.
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=128,
        ),

        # основной текст (content)
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),

        # embeddings по content
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=config.DIM,
        ),
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),

        # embeddings по summary (ТОЛЬКО для H2)
        FieldSchema(
            name="dense_vector_summary",
            dtype=DataType.FLOAT_VECTOR,
            dim=config.DIM,
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description="Documents / sections / subsections",
        enable_dynamic_field=True,
    )



def _create_indexes(collection: Collection):
    logger.info("🔧 Создаём индексы...")

    # content
    collection.create_index(
        field_name="dense_vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 32, "efConstruction": 200},
        },
    )

    collection.create_index(
        field_name="sparse_vector",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        },
    )

    # summary (H2)
    collection.create_index(
        field_name="dense_vector_summary",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 32, "efConstruction": 200},
        },
    )



def init_milvus() -> Collection:
    """
    Подключение + проверка/создание коллекции.
    Без пересоздания при рестарте.
    """
    logger.info(
        f"🔌 Подключаемся к Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}"
    )
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
    )

    if utility.has_collection(config.COLLECTION_NAME):
        logger.info(
            f"✅ Коллекция '{config.COLLECTION_NAME}' уже существует"
        )
        collection = Collection(config.COLLECTION_NAME)
        collection.load()
        return collection

    logger.warning(
        f"⚠ Коллекция '{config.COLLECTION_NAME}' отсутствует, создаём новую..."
    )

    schema = _build_schema()
    collection = Collection(
        name=config.COLLECTION_NAME,
        schema=schema,
    )

    _create_indexes(collection)
    collection.load()

    logger.info(f"✅ Коллекция '{config.COLLECTION_NAME}' создана и загружена")
    return collection

def get_collection() -> Collection:
    """
    Возвращает объект коллекции Milvus (должен быть уже init_milvus).
    """
    return Collection(name=config.COLLECTION_NAME)
    
if __name__ == "__main__":
    init_milvus()
