"""RAG 서비스 및 API 테스트"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.rag_service import RAGService, DocumentChunk

client = TestClient(app)


@pytest.fixture
def rag_service():
    """RAG 서비스 인스턴스"""
    return RAGService()


@pytest.fixture
def sample_text():
    """샘플 텍스트"""
    return """
    머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 자동으로 학습하는 능력을 갖게 하는 기술입니다.
    딥러닝은 머신러닝의 한 종류로, 신경망을 이용하여 복잡한 패턴을 학습합니다.
    Python은 머신러닝과 데이터 과학에서 가장 인기있는 프로그래밍 언어입니다.
    """


# ========================================
# DocumentChunk 테스트
# ========================================


class TestDocumentChunk:
    """DocumentChunk 테스트"""

    def test_chunk_creation(self):
        """청크 생성"""
        chunk = DocumentChunk("테스트 내용", {"source": "test"})
        assert chunk.content == "테스트 내용"
        assert chunk.metadata["source"] == "test"

    def test_chunk_to_dict(self):
        """청크 딕셔너리 변환"""
        chunk = DocumentChunk("테스트", {"source": "test"})
        result = chunk.to_dict()

        assert "content" in result
        assert "metadata" in result
        assert result["content"] == "테스트"


# ========================================
# RAGService 단위 테스트
# ========================================


class TestRAGService:
    """RAG 서비스 테스트"""

    def test_initialization(self, rag_service):
        """초기화"""
        assert rag_service.documents == []
        assert rag_service.embeddings is None

    def test_create_chunks(self, rag_service, sample_text):
        """청크 생성"""
        chunks = rag_service._create_chunks(sample_text, "test_source")

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.metadata["source"] == "test_source" for c in chunks)

    def test_load_text(self, rag_service, sample_text):
        """텍스트 로드"""
        result = rag_service.load_text(sample_text)

        assert result["status"] == "success"
        assert result["chunks_created"] > 0

    def test_get_documents_info_empty(self, rag_service):
        """문서 정보 - 빈 상태"""
        result = rag_service.get_documents_info()

        assert result["status"] == "success"
        assert result["total_documents"] == 0

    def test_get_documents_info_with_docs(self, rag_service, sample_text):
        """문서 정보 - 문서 있을 때"""
        rag_service.load_text(sample_text)
        result = rag_service.get_documents_info()

        assert result["status"] == "success"
        assert result["total_documents"] > 0

    def test_configure_rag(self, rag_service):
        """RAG 설정"""
        result = rag_service.configure_rag(
            chunk_size=256,
            chunk_overlap=32,
            top_k=3,
        )

        assert result["status"] == "success"
        assert rag_service.chunk_size == 256
        assert rag_service.chunk_overlap == 32
        assert rag_service.top_k == 3

    def test_get_rag_config(self, rag_service):
        """RAG 설정 조회"""
        rag_service.configure_rag(chunk_size=256)
        result = rag_service.get_rag_config()

        assert result["status"] == "success"
        assert result["chunk_size"] == 256

    def test_clear_documents(self, rag_service, sample_text):
        """문서 초기화"""
        rag_service.load_text(sample_text)
        assert len(rag_service.documents) > 0

        result = rag_service.clear_documents()
        assert result["status"] == "success"
        assert len(rag_service.documents) == 0

    def test_conversation_history(self, rag_service):
        """대화 히스토리"""
        result = rag_service.get_conversation_history()

        assert result["status"] == "success"
        assert result["total_messages"] == 0

    def test_clear_conversation_history(self, rag_service):
        """대화 히스토리 초기화"""
        rag_service.conversation_history.append({"role": "user", "content": "test"})
        assert len(rag_service.conversation_history) == 1

        result = rag_service.clear_conversation_history()
        assert result["status"] == "success"
        assert len(rag_service.conversation_history) == 0

    def test_get_statistics_empty(self, rag_service):
        """통계 - 빈 상태"""
        result = rag_service.get_statistics()

        assert result["status"] == "success"
        assert result["total_documents"] == 0

    def test_get_statistics_with_docs(self, rag_service, sample_text):
        """통계 - 문서 있을 때"""
        rag_service.load_text(sample_text)
        result = rag_service.get_statistics()

        assert result["status"] == "success"
        assert result["total_documents"] > 0


# ========================================
# API 엔드포인트 테스트
# ========================================


class TestRAGAPI:
    """RAG API 테스트"""

    def test_rag_health(self):
        """헬스 체크"""
        response = client.get("/rag/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_initialize_rag(self):
        """초기화"""
        response = client.post("/rag/initialize")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_load_text(self):
        """텍스트 로드"""
        response = client.post(
            "/rag/load-text",
            json={"text": "테스트 문서 입니다. 이것은 샘플 텍스트입니다."},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_get_documents_info(self):
        """문서 정보 조회"""
        response = client.get("/rag/documents/info")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_configure_rag(self):
        """RAG 설정"""
        response = client.post(
            "/rag/config",
            json={
                "chunk_size": 256,
                "chunk_overlap": 32,
                "top_k": 3,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_get_rag_config(self):
        """RAG 설정 조회"""
        response = client.get("/rag/config")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_conversation_history(self):
        """대화 히스토리 조회"""
        response = client.get("/rag/history")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_clear_conversation_history(self):
        """대화 히스토리 초기화"""
        response = client.post("/rag/history/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_get_statistics(self):
        """통계 조회"""
        response = client.get("/rag/statistics")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_clear_documents(self):
        """문서 초기화"""
        response = client.post("/rag/documents/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
