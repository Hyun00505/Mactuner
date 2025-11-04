"""학습 엔진 테스트"""
import pytest
import torch
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.training_service import TrainingService
from backend.utils.mac_optimization import MACOptimizer

client = TestClient(app)


# ========================================
# 픽스처
# ========================================


@pytest.fixture
def training_service():
    """학습 서비스 인스턴스"""
    return TrainingService()


@pytest.fixture
def mock_model():
    """Mock 모델"""
    # 간단한 선형 모델 생성
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
    )
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock 토크나이저"""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except:
        return None


# ========================================
# TrainingService 단위 테스트
# ========================================


class TestTrainingService:
    """학습 서비스 테스트"""

    def test_initialization(self, training_service):
        """초기화 테스트"""
        assert training_service.model is None
        assert training_service.tokenizer is None
        assert training_service.trainer is None
        assert training_service.training_args is None
        assert training_service.lora_config is None

    def test_setup_lora(self, training_service):
        """LoRA 설정 테스트"""
        result = training_service.setup_lora(
            rank=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj"]
        )

        assert result["status"] == "configured"
        assert result["rank"] == 8
        assert result["alpha"] == 16
        assert result["dropout"] == 0.1

    def test_setup_qlora(self, training_service):
        """QLoRA 설정 테스트"""
        result = training_service.setup_qlora(
            rank=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj"]
        )

        assert result["status"] == "configured"
        assert result["quantization"] == "4-bit"
        assert result["rank"] == 8

    def test_configure_training_args(self, training_service, tmp_path):
        """TrainingArguments 설정 테스트"""
        output_dir = str(tmp_path / "checkpoints")

        result = training_service.configure_training_args(
            output_dir=output_dir,
            num_epochs=3,
            batch_size=4,
            learning_rate=5e-5,
            warmup_steps=500,
        )

        assert result["status"] == "configured"
        assert result["num_epochs"] == 3
        assert result["batch_size"] == 4
        assert result["learning_rate"] == 5e-5

    def test_recommend_parameters_small_model(self, training_service):
        """파라미터 추천 - 소형 모델"""
        result = training_service.recommend_parameters(
            model_size_params=100e6, dataset_size=5000
        )

        assert "recommendations" in result
        assert "batch_size" in result["recommendations"]
        assert "learning_rate" in result["recommendations"]

    def test_recommend_parameters_medium_model(self, training_service):
        """파라미터 추천 - 중형 모델"""
        result = training_service.recommend_parameters(
            model_size_params=3e9, dataset_size=10000
        )

        assert result["recommendations"]["use_qlora"] == True
        assert result["recommendations"]["lora_rank"] == 16

    def test_recommend_parameters_large_model(self, training_service):
        """파라미터 추천 - 대형 모델"""
        result = training_service.recommend_parameters(
            model_size_params=13e9, dataset_size=50000
        )

        assert result["recommendations"]["use_qlora"] == True
        assert result["recommendations"]["lora_rank"] == 8

    def test_get_training_status_not_started(self, training_service):
        """학습 상태 조회 - 미시작"""
        status = training_service.get_training_status()
        assert status["status"] == "not_started"

    def test_get_training_history_empty(self, training_service):
        """학습 이력 조회 - 빈 상태"""
        history = training_service.get_training_history()
        assert isinstance(history, dict)

    def test_setup_lora_defaults(self, training_service):
        """LoRA 기본값 설정"""
        result = training_service.setup_lora()
        assert result["rank"] == 8
        assert result["alpha"] == 16
        assert result["dropout"] == 0.1

    def test_setup_qlora_defaults(self, training_service):
        """QLoRA 기본값 설정"""
        result = training_service.setup_qlora()
        assert result["rank"] == 8
        assert result["quantization"] == "4-bit"


# ========================================
# API 엔드포인트 테스트
# ========================================


class TestTrainingAPI:
    """학습 API 테스트"""

    def test_training_health(self):
        """헬스 체크"""
        response = client.get("/train/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_config_lora(self):
        """LoRA 설정 API"""
        response = client.post(
            "/train/config-lora",
            json={
                "rank": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["config"]["rank"] == 16

    def test_config_qlora(self):
        """QLoRA 설정 API"""
        response = client.post(
            "/train/config-qlora",
            json={
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["config"]["quantization"] == "4-bit"

    def test_config_training_args(self, tmp_path):
        """TrainingArguments 설정 API"""
        output_dir = str(tmp_path / "checkpoints")

        response = client.post(
            "/train/config-training-args",
            json={
                "output_dir": output_dir,
                "num_epochs": 5,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_recommend_parameters(self):
        """파라미터 추천 API"""
        response = client.post(
            "/train/recommend-parameters",
            json={"model_size_params": 7000000000, "dataset_size": 10000},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "recommendations" in data["data"]

    def test_get_training_status(self):
        """학습 상태 조회 API"""
        response = client.get("/train/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_training_history(self):
        """학습 이력 조회 API"""
        response = client.get("/train/history")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_prepare_dataset(self):
        """데이터셋 준비 API"""
        response = client.post(
            "/train/prepare-dataset",
            json={
                "text_column": "text",
                "label_column": "label",
                "max_length": 512,
                "test_size": 0.1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_start_training_without_preparation(self):
        """학습 시작 - 준비 없음"""
        response = client.post("/train/start")
        assert response.status_code == 400

    def test_save_model_without_model(self, tmp_path):
        """모델 저장 - 모델 없음"""
        output_dir = str(tmp_path / "model")
        response = client.post(
            "/train/save",
            params={"output_dir": output_dir},
        )

        assert response.status_code == 400

    def test_evaluate_without_trainer(self):
        """모델 평가 - Trainer 없음"""
        response = client.post("/train/evaluate")
        assert response.status_code == 400


# ========================================
# 파라미터 추천 테스트
# ========================================


class TestParameterRecommendations:
    """파라미터 추천 테스트"""

    def test_small_model_small_dataset(self, training_service):
        """소형 모델 + 소형 데이터셋"""
        result = training_service.recommend_parameters(
            model_size_params=100e6, dataset_size=100
        )

        recs = result["recommendations"]
        assert recs["learning_rate"] == 1e-4
        assert recs["num_epochs"] == 10

    def test_medium_model_medium_dataset(self, training_service):
        """중형 모델 + 중형 데이터셋"""
        result = training_service.recommend_parameters(
            model_size_params=3e9, dataset_size=5000
        )

        recs = result["recommendations"]
        assert recs["use_qlora"] == True
        assert recs["num_epochs"] == 5

    def test_large_model_large_dataset(self, training_service):
        """대형 모델 + 대형 데이터셋"""
        result = training_service.recommend_parameters(
            model_size_params=13e9, dataset_size=100000
        )

        recs = result["recommendations"]
        assert recs["use_qlora"] == True
        assert recs["learning_rate"] == 2e-5
        assert recs["num_epochs"] == 3


# ========================================
# 에러 처리 테스트
# ========================================


class TestErrorHandling:
    """에러 처리 테스트"""

    def test_config_training_args_invalid_epochs(self):
        """잘못된 에포크 수"""
        response = client.post(
            "/train/config-training-args",
            json={
                "num_epochs": 0,
                "batch_size": 4,
                "learning_rate": 5e-5,
            },
        )

        # Validation error
        assert response.status_code == 422

    def test_recommend_parameters_zero_model_size(self, training_service):
        """모델 크기 0"""
        result = training_service.recommend_parameters(
            model_size_params=0, dataset_size=1000
        )

        # 여전히 추천값 반환
        assert "recommendations" in result

    def test_lora_config_invalid_rank(self):
        """잘못된 LoRA rank"""
        response = client.post(
            "/train/config-lora",
            json={
                "rank": -1,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj"],
            },
        )

        # 형식은 정확하지만 양수가 아님
        assert response.status_code == 200  # Pydantic은 정수 검증만 함


# ========================================
# 성능 테스트
# ========================================


class TestPerformance:
    """성능 테스트"""

    def test_parameter_recommendation_performance(self, training_service):
        """파라미터 추천 성능"""
        import time

        start = time.time()
        for _ in range(100):
            training_service.recommend_parameters(
                model_size_params=7e9, dataset_size=10000
            )
        elapsed = time.time() - start

        # 100회 추천이 0.5초 이내
        assert elapsed < 0.5

    def test_lora_setup_performance(self, training_service):
        """LoRA 설정 성능"""
        import time

        start = time.time()
        for _ in range(50):
            training_service.setup_lora()
        elapsed = time.time() - start

        # 50회 설정이 0.2초 이내
        assert elapsed < 0.2

    def test_training_args_configuration_performance(self, training_service, tmp_path):
        """TrainingArguments 설정 성능"""
        import time

        start = time.time()
        for _ in range(20):
            training_service.configure_training_args(
                output_dir=str(tmp_path), num_epochs=3
            )
        elapsed = time.time() - start

        # 20회 설정이 1초 이내
        assert elapsed < 1.0


# ========================================
# 통합 테스트
# ========================================


class TestIntegration:
    """통합 테스트"""

    def test_lora_then_training_args(self, training_service, tmp_path):
        """LoRA 설정 후 TrainingArguments 설정"""
        # LoRA 설정
        lora_result = training_service.setup_lora(rank=16, alpha=32)
        assert lora_result["status"] == "configured"

        # TrainingArguments 설정
        args_result = training_service.configure_training_args(
            output_dir=str(tmp_path), num_epochs=5
        )
        assert args_result["status"] == "configured"

        # 모두 설정됨
        assert training_service.lora_config is not None
        assert training_service.training_args is not None

    def test_qlora_then_recommendation(self, training_service):
        """QLoRA 설정 후 파라미터 추천"""
        # QLoRA 설정
        qlora_result = training_service.setup_qlora(rank=8, alpha=16)
        assert qlora_result["status"] == "configured"
        assert qlora_result["quantization"] == "4-bit"

        # 파라미터 추천
        rec_result = training_service.recommend_parameters(
            model_size_params=13e9, dataset_size=50000
        )
        assert "recommendations" in rec_result


# ========================================
# 설정 검증 테스트
# ========================================


class TestConfigValidation:
    """설정 검증 테스트"""

    def test_lora_config_values(self, training_service):
        """LoRA 설정값 검증"""
        config = training_service.setup_lora(rank=32, alpha=64, dropout=0.2)

        assert config["rank"] == 32
        assert config["alpha"] == 64
        assert config["dropout"] == 0.2

    def test_training_args_defaults(self, training_service, tmp_path):
        """TrainingArguments 기본값"""
        result = training_service.configure_training_args(output_dir=str(tmp_path))

        assert result["num_epochs"] == 3
        assert result["batch_size"] == 4
        assert result["learning_rate"] == 5e-5

    def test_recommend_learning_rates(self, training_service):
        """추천 학습률 검증"""
        # 소형 데이터: 높은 학습률
        small = training_service.recommend_parameters(1e9, 100)
        assert small["recommendations"]["learning_rate"] == 1e-4

        # 중형 데이터: 중간 학습률
        medium = training_service.recommend_parameters(3e9, 5000)
        assert medium["recommendations"]["learning_rate"] == 5e-5

        # 대형 데이터: 낮은 학습률
        large = training_service.recommend_parameters(13e9, 100000)
        assert large["recommendations"]["learning_rate"] == 2e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
