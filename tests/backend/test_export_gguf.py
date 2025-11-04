"""GGUF 변환 및 배포 테스트"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.quantization_service import QuantizationService

client = TestClient(app)


@pytest.fixture
def quantization_service():
    """양자화 서비스 인스턴스"""
    return QuantizationService()


# ========================================
# QuantizationService 단위 테스트
# ========================================


class TestQuantizationService:
    """양자화 서비스 테스트"""

    def test_initialization(self, quantization_service):
        """초기화 테스트"""
        assert quantization_service.device is not None
        assert len(quantization_service.QUANTIZATION_METHODS) == 10

    def test_get_available_methods(self, quantization_service):
        """지원되는 양자화 방식 조회"""
        result = quantization_service.get_available_methods()

        assert result["status"] == "success"
        assert len(result["available_methods"]) == 10
        assert "Q4_K" in result["available_methods"]
        assert result["recommended_method"] == "Q4_K"

    def test_quantization_methods_structure(self, quantization_service):
        """양자화 방식 구조 검증"""
        methods = quantization_service.QUANTIZATION_METHODS

        for method_name, method_info in methods.items():
            assert "description" in method_info
            assert "quality" in method_info
            assert "size" in method_info
            assert 1 <= method_info["quality"] <= 10
            assert 1 <= method_info["size"] <= 7

    def test_get_recommended_method_small_model(self, quantization_service):
        """권장 방식 - 소형 모델"""
        result = quantization_service.get_recommended_method(model_size_gb=2.0)

        assert result["status"] == "success"
        assert result["model_size_gb"] == 2.0
        assert result["primary_recommendation"] == "Q6_K"
        assert len(result["all_recommendations"]) >= 2

    def test_get_recommended_method_medium_model(self, quantization_service):
        """권장 방식 - 중형 모델"""
        result = quantization_service.get_recommended_method(model_size_gb=7.0)

        assert result["status"] == "success"
        assert result["primary_recommendation"] == "Q5_K"

    def test_get_recommended_method_large_model(self, quantization_service):
        """권장 방식 - 대형 모델"""
        result = quantization_service.get_recommended_method(model_size_gb=20.0)

        assert result["status"] == "success"
        assert result["primary_recommendation"] == "Q4_0"

    def test_get_recommended_method_very_large_model(self, quantization_service):
        """권장 방식 - 초대형 모델"""
        result = quantization_service.get_recommended_method(model_size_gb=60.0)

        assert result["status"] == "success"
        assert result["primary_recommendation"] == "Q2_K"

    def test_conversion_history_empty(self, quantization_service):
        """변환 이력 - 빈 상태"""
        result = quantization_service.get_conversion_history()

        assert result["status"] == "success"
        assert result["total_conversions"] == 0
        assert result["history"] == []

    def test_clear_conversion_history(self, quantization_service):
        """변환 이력 초기화"""
        result = quantization_service.clear_conversion_history()

        assert result["status"] == "success"
        assert len(quantization_service.conversion_history) == 0

    def test_compression_statistics_empty(self, quantization_service):
        """압축 통계 - 빈 상태"""
        result = quantization_service.get_compression_statistics()

        assert result["status"] == "success"
        assert result["total_conversions"] == 0


# ========================================
# API 엔드포인트 테스트
# ========================================


class TestGGUFAPI:
    """GGUF API 테스트"""

    def test_gguf_health(self):
        """헬스 체크"""
        response = client.get("/gguf/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["service"] == "gguf_export"

    def test_get_quantization_methods(self):
        """양자화 방식 조회"""
        response = client.get("/gguf/methods")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["status"] == "success"
        assert data["total_methods"] == 10
        assert "available_methods" in data

    def test_get_recommended_methods_small(self):
        """권장 방식 조회 - 소형 모델"""
        response = client.get("/gguf/methods/recommended?model_size_gb=3.0")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["status"] == "success"
        assert data["model_size_gb"] == 3.0
        assert "primary_recommendation" in data

    def test_get_recommended_methods_medium(self):
        """권장 방식 조회 - 중형 모델"""
        response = client.get("/gguf/methods/recommended?model_size_gb=10.0")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["primary_recommendation"] == "Q4_K"

    def test_get_recommended_methods_large(self):
        """권장 방식 조회 - 대형 모델"""
        response = client.get("/gguf/methods/recommended?model_size_gb=40.0")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["primary_recommendation"] == "Q3_K"

    def test_get_conversion_history(self):
        """변환 이력 조회"""
        response = client.get("/gguf/history")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["status"] == "success"
        assert "history" in data

    def test_clear_conversion_history(self):
        """변환 이력 초기화"""
        response = client.post("/gguf/history/clear")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["status"] == "success"

    def test_get_compression_statistics(self):
        """압축 통계 조회"""
        response = client.get("/gguf/statistics")
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["status"] == "success"

    def test_convert_to_gguf_invalid_method(self):
        """GGUF 변환 - 잘못된 방식"""
        response = client.post(
            "/gguf/convert",
            json={
                "model_path": "/path/to/model",
                "output_dir": "/path/to/output",
                "quantization_method": "INVALID",
                "use_gpu": True,
            },
        )

        assert response.status_code == 400

    def test_validate_gguf_invalid_path(self):
        """GGUF 검증 - 잘못된 경로"""
        response = client.post(
            "/gguf/validate",
            json={"gguf_path": "/nonexistent/path/model.gguf"},
        )

        assert response.status_code == 400


# ========================================
# 양자화 방식 검증 테스트
# ========================================


class TestQuantizationMethodsValidation:
    """양자화 방식 검증 테스트"""

    def test_all_methods_have_required_fields(self, quantization_service):
        """모든 방식이 필수 필드를 가짐"""
        for method, info in quantization_service.QUANTIZATION_METHODS.items():
            assert "description" in info
            assert "quality" in info
            assert "size" in info
            assert isinstance(info["quality"], int)
            assert isinstance(info["size"], int)

    def test_quality_levels_ordered(self, quantization_service):
        """품질 레벨이 올바르게 순서됨"""
        methods = [
            ("Q2_K", 1),
            ("Q3_K", 2),
            ("Q4_0", 3),
            ("Q4_K", 4),
            ("Q5_0", 5),
            ("Q5_K", 6),
            ("Q6_K", 7),
            ("Q8_0", 8),
            ("F16", 9),
            ("F32", 10),
        ]

        for method, expected_quality in methods:
            actual_quality = quantization_service.QUANTIZATION_METHODS[method]["quality"]
            assert actual_quality == expected_quality

    def test_size_levels_ordered(self, quantization_service):
        """크기 레벨이 올바르게 순서됨"""
        for method, info in quantization_service.QUANTIZATION_METHODS.items():
            # 사이즈가 1-7 범위 내
            assert 1 <= info["size"] <= 7


# ========================================
# 권장 방식 로직 테스트
# ========================================


class TestRecommendationLogic:
    """권장 방식 로직 테스트"""

    def test_recommendation_coverage(self, quantization_service):
        """모든 크기 범위에 권장 방식 존재"""
        sizes = [1, 5, 10, 20, 35, 50, 70]

        for size in sizes:
            result = quantization_service.get_recommended_method(size)
            assert result["primary_recommendation"] in quantization_service.QUANTIZATION_METHODS

    def test_recommendation_has_alternatives(self, quantization_service):
        """권장 방식에 대안이 존재"""
        result = quantization_service.get_recommended_method(10.0)

        assert len(result["all_recommendations"]) >= 2
        assert result["all_recommendations"][0]["rank"] == 1
        assert result["all_recommendations"][1]["rank"] == 2


# ========================================
# 성능 테스트
# ========================================


class TestPerformance:
    """성능 테스트"""

    def test_get_methods_performance(self):
        """양자화 방식 조회 성능"""
        import time

        start = time.time()
        for _ in range(100):
            client.get("/gguf/methods")
        elapsed = time.time() - start

        # 100회 조회 < 1초
        assert elapsed < 1.0

    def test_recommendation_performance(self):
        """권장 방식 조회 성능"""
        import time

        start = time.time()
        for _ in range(100):
            client.get("/gguf/methods/recommended?model_size_gb=10.0")
        elapsed = time.time() - start

        # 100회 조회 < 1초
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
