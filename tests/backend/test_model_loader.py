"""모델 로더 테스트"""
import pytest
import torch
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.model_service import ModelService
from backend.utils.mac_optimization import MACOptimizer

client = TestClient(app)


class TestMACOptimizer:
    """MAC 최적화 테스트"""

    def test_get_device(self):
        """디바이스 감지 테스트"""
        device = MACOptimizer.get_device()
        assert device.type in ["mps", "cuda", "cpu"]

    def test_get_memory_stats(self):
        """메모리 통계 조회 테스트"""
        stats = MACOptimizer.get_memory_stats()
        assert "total_gb" in stats
        assert "available_gb" in stats
        assert "used_gb" in stats
        assert "percent" in stats
        assert "device" in stats
        assert stats["total_gb"] > 0

    def test_get_optimal_batch_size(self):
        """최적 배치 크기 계산 테스트"""
        # 소형 모델
        batch_size_small = MACOptimizer.get_optimal_batch_size(100e6)
        assert batch_size_small > 0

        # 중형 모델
        batch_size_medium = MACOptimizer.get_optimal_batch_size(3e9)
        assert batch_size_medium > 0

        # 대형 모델
        batch_size_large = MACOptimizer.get_optimal_batch_size(13e9)
        assert batch_size_large > 0

        # 배치 크기 순서 확인
        assert batch_size_small >= batch_size_medium >= batch_size_large

    def test_estimate_model_memory(self):
        """모델 메모리 추정 테스트"""
        memory = MACOptimizer.estimate_model_memory(1e9)
        assert memory > 0
        assert memory < 10  # 1B 모델은 10GB 미만


class TestModelService:
    """모델 서비스 테스트"""

    @pytest.fixture
    def service(self):
        """모델 서비스 인스턴스"""
        return ModelService()

    def test_service_initialization(self, service):
        """서비스 초기화 테스트"""
        assert service.device is not None
        assert service.cache_dir is not None

    def test_list_local_models_empty(self, service):
        """로컬 모델 목록 조회 - 빈 상태"""
        models = service.list_local_models()
        assert isinstance(models, list)


class TestModelLoaderAPI:
    """모델 로더 API 테스트"""

    def test_model_health(self):
        """모델 헬스 체크"""
        response = client.get("/model/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_get_current_model_no_model_loaded(self):
        """현재 모델 조회 - 로드된 모델 없음"""
        response = client.get("/model/current")
        # 모델이 없으면 404 반환
        assert response.status_code == 404

    def test_list_local_models(self):
        """로컬 모델 목록 조회"""
        response = client.get("/model/local-models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert isinstance(data["models"], list)
        assert isinstance(data["count"], int)

    def test_unload_model_no_model(self):
        """모델 언로드 - 로드된 모델 없음"""
        response = client.post("/model/unload")
        assert response.status_code == 404

    def test_get_model_info_error(self):
        """모델 정보 조회 - 오류"""
        response = client.get("/model/info/nonexistent-model")
        # 토큰이 없으면 오류 발생
        assert response.status_code in [400, 403]


class TestModelServiceIntegration:
    """모델 서비스 통합 테스트"""

    @pytest.fixture
    def service(self):
        return ModelService()

    def test_metadata_extraction(self, service):
        """메타데이터 추출 테스트"""
        # GPT-2 모델은 비교적 작으므로 테스트에 유용
        # (실제 테스트에서는 토큰이 필요할 수 있음)
        try:
            model, tokenizer, metadata = service.load_from_hub("gpt2")

            # 메타데이터 검증
            assert "model_id" in metadata
            assert "source" in metadata
            assert "num_parameters" in metadata
            assert "estimated_memory_gb" in metadata
            assert "device" in metadata
            assert metadata["source"] == "hub"
            assert metadata["num_parameters"] > 0

            # 모델 정보 검증
            assert model is not None
            assert tokenizer is not None

        except RuntimeError as e:
            # 네트워크 오류는 무시
            pytest.skip(f"네트워크 오류: {str(e)}")


class TestErrorHandling:
    """에러 처리 테스트"""

    def test_load_invalid_model(self):
        """존재하지 않는 모델 로드"""
        service = ModelService()
        with pytest.raises(RuntimeError):
            service.load_from_hub("definitely-nonexistent-model-12345")

    def test_load_local_invalid_path(self):
        """존재하지 않는 로컬 경로"""
        service = ModelService()
        with pytest.raises(FileNotFoundError):
            service.load_local("/nonexistent/path/to/model")


# ========================================
# 성능 테스트
# ========================================


class TestPerformance:
    """성능 테스트"""

    def test_memory_stats_performance(self):
        """메모리 통계 조회 성능"""
        import time

        start = time.time()
        for _ in range(100):
            MACOptimizer.get_memory_stats()
        elapsed = time.time() - start

        # 100회 조회가 1초 이내에 완료되어야 함
        assert elapsed < 1.0

    def test_optimal_batch_size_performance(self):
        """배치 크기 계산 성능"""
        import time

        start = time.time()
        for _ in range(1000):
            MACOptimizer.get_optimal_batch_size(7e9)
        elapsed = time.time() - start

        # 1000회 계산이 100ms 이내에 완료되어야 함
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
