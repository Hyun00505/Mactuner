"""MacTuner 설정 관리 모듈"""
import os
from pathlib import Path

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Settings:
    """MacTuner 전역 설정"""

    # ========================================
    # API 설정
    # ========================================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "True").lower() == "true"

    # ========================================
    # Hugging Face 설정
    # ========================================
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    MODEL_CACHE_DIR: str = os.path.expanduser(
        os.getenv("MODEL_CACHE_DIR", "~/.cache/huggingface/hub")
    )

    # ========================================
    # 경로 설정
    # ========================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "output")
    LOG_DIR: Path = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")

    # ========================================
    # 학습 기본값
    # ========================================
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "4"))
    DEFAULT_LEARNING_RATE: float = float(os.getenv("DEFAULT_LEARNING_RATE", "5e-5"))
    DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "3"))
    DEFAULT_WARMUP_STEPS: int = int(os.getenv("DEFAULT_WARMUP_STEPS", "500"))

    # ========================================
    # LoRA 설정
    # ========================================
    LORA_RANK: int = int(os.getenv("LORA_RANK", "8"))
    LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", "16"))
    LORA_DROPOUT: float = float(os.getenv("LORA_DROPOUT", "0.1"))

    # ========================================
    # RAG 설정
    # ========================================
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    RAG_EMBEDDING_MODEL: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

    # ========================================
    # 로깅 설정
    # ========================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ========================================
    # 성능 최적화
    # ========================================
    USE_MAC_MPS: bool = os.getenv("USE_MAC_MPS", "True").lower() == "true"
    GRADIENT_CHECKPOINTING: bool = os.getenv("GRADIENT_CHECKPOINTING", "True").lower() == "true"
    MIXED_PRECISION: str = os.getenv("MIXED_PRECISION", "auto")

    # ========================================
    # 환경 설정
    # ========================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    def __init__(self):
        """설정 초기화 - 필요한 디렉토리 생성"""
        for dir_path in [self.DATA_DIR, self.OUTPUT_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# 전역 설정 인스턴스
settings = Settings()
