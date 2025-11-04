"""데이터셋 도구 테스트"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.dataset_service import DatasetService

client = TestClient(app)


# ========================================
# 픽스처
# ========================================


@pytest.fixture
def sample_csv():
    """샘플 CSV 데이터"""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "text": ["hello world", "test data", "sample text", "more data", "final entry"],
            "label": ["A", "B", "A", "C", "B"],
            "score": [0.8, 0.9, 0.7, 0.6, 0.85],
        }
    )
    return df.to_csv(index=False).encode()


@pytest.fixture
def sample_csv_with_missing():
    """결측치가 있는 샘플 CSV"""
    df = pd.DataFrame(
        {
            "id": [1, 2, None, 4, 5],
            "text": ["hello", None, "test", "data", "text"],
            "label": ["A", "B", "A", None, "B"],
        }
    )
    return df.to_csv(index=False).encode()


@pytest.fixture
def sample_json():
    """샘플 JSON 데이터"""
    data = [
        {"id": 1, "text": "hello", "label": "A"},
        {"id": 2, "text": "world", "label": "B"},
        {"id": 3, "text": "test", "label": "A"},
    ]
    return pd.DataFrame(data).to_json(orient="records").encode()


@pytest.fixture
def dataset_service():
    """데이터셋 서비스 인스턴스"""
    return DatasetService()


# ========================================
# DatasetService 단위 테스트
# ========================================


class TestDatasetService:
    """데이터셋 서비스 테스트"""

    def test_initialization(self, dataset_service):
        """초기화 테스트"""
        assert dataset_service.data is None
        assert dataset_service.original_data is None
        assert dataset_service.file_info == {}

    def test_load_csv(self, dataset_service, sample_csv):
        """CSV 로드 테스트"""
        df, file_info = dataset_service.load_dataset(sample_csv, "test.csv")

        assert len(df) == 5
        assert len(df.columns) == 4
        assert file_info["filename"] == "test.csv"
        assert file_info["rows"] == 5

    def test_load_json(self, dataset_service, sample_json):
        """JSON 로드 테스트"""
        df, file_info = dataset_service.load_dataset(sample_json, "test.json")

        assert len(df) == 3
        assert file_info["filename"] == "test.json"

    def test_get_data_info(self, dataset_service, sample_csv):
        """데이터 정보 조회 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        info = dataset_service.get_data_info()

        assert info["shape"]["rows"] == 5
        assert info["shape"]["columns"] == 4
        assert "size_mb" in info
        assert "dtypes" in info

    def test_get_preview(self, dataset_service, sample_csv):
        """데이터 미리보기 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        preview = dataset_service.get_preview(n_rows=2)

        assert len(preview["head"]) == 2
        assert len(preview["tail"]) == 2
        assert preview["total_rows"] == 5

    # ========================================
    # 데이터 정제 테스트
    # ========================================

    def test_handle_missing_values_drop(
        self, dataset_service, sample_csv_with_missing
    ):
        """결측치 제거 테스트"""
        dataset_service.load_dataset(sample_csv_with_missing, "test.csv")
        result = dataset_service.handle_missing_values(strategy="drop")

        assert result["strategy"] == "drop"
        assert result["missing_before"] > 0
        assert result["missing_after"] == 0

    def test_handle_missing_values_fill(
        self, dataset_service, sample_csv_with_missing
    ):
        """결측치 채우기 테스트"""
        dataset_service.load_dataset(sample_csv_with_missing, "test.csv")
        result = dataset_service.handle_missing_values(strategy="fill", value=0)

        assert result["missing_before"] > 0

    def test_remove_duplicates(self, dataset_service):
        """중복 제거 테스트"""
        df = pd.DataFrame(
            {"id": [1, 2, 2, 3], "text": ["a", "b", "b", "c"]}
        )
        csv_data = df.to_csv(index=False).encode()

        dataset_service.load_dataset(csv_data, "test.csv")
        result = dataset_service.remove_duplicates()

        assert result["duplicates_removed"] == 1
        assert result["rows_after"] == 3

    def test_normalize_text(self, dataset_service):
        """텍스트 정규화 테스트"""
        df = pd.DataFrame(
            {"text": ["HELLO World!!", "TEST@#$", "DATA   123"]}
        )
        csv_data = df.to_csv(index=False).encode()

        dataset_service.load_dataset(csv_data, "test.csv")
        result = dataset_service.normalize_text(columns=["text"])

        assert "text" in result["normalized_columns"]
        # 정규화 확인
        assert dataset_service.data["text"].iloc[0] == "hello world"

    def test_filter_by_text_length(self, dataset_service, sample_csv):
        """텍스트 길이 필터링 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        result = dataset_service.filter_by_text_length(
            column="text", min_length=5, max_length=15
        )

        assert result["rows_before"] == 5
        assert result["rows_removed"] >= 0

    def test_analyze_token_length(self, dataset_service, sample_csv):
        """토큰 길이 분석 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        result = dataset_service.analyze_token_length(column="text", model_name="gpt2")

        assert "min_tokens" in result
        assert "max_tokens" in result
        assert "mean_tokens" in result
        assert result["model"] == "gpt2"
        assert result["min_tokens"] >= 0

    # ========================================
    # EDA 테스트
    # ========================================

    def test_get_statistics(self, dataset_service, sample_csv):
        """통계 조회 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        stats = dataset_service.get_statistics()

        assert stats["total_rows"] == 5
        assert stats["total_columns"] == 4
        assert "memory_mb" in stats
        assert "dtypes" in stats

    def test_get_missing_values(self, dataset_service, sample_csv_with_missing):
        """결측치 분석 테스트"""
        dataset_service.load_dataset(sample_csv_with_missing, "test.csv")
        missing = dataset_service.get_missing_values()

        assert "columns_with_missing" in missing
        assert "missing_percentage" in missing
        assert missing["total_missing_values"] > 0

    def test_get_value_distribution(self, dataset_service, sample_csv):
        """값 분포 분석 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        dist = dataset_service.get_value_distribution(column="label", top_n=5)

        assert dist["unique_values"] >= 1
        assert "top_values" in dist

    def test_get_correlation(self, dataset_service, sample_csv):
        """상관관계 분석 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        corr = dataset_service.get_correlation()

        assert "correlation" in corr or "message" in corr

    # ========================================
    # 데이터 분할 테스트
    # ========================================

    def test_train_test_split(self, dataset_service, sample_csv):
        """Train/Test 분할 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        train_df, test_df = dataset_service.train_test_split(test_size=0.2)

        assert len(train_df) + len(test_df) == 5
        assert len(test_df) == 1  # 5 * 0.2 = 1
        assert len(train_df) == 4

    def test_train_test_split_invalid_size(self, dataset_service, sample_csv):
        """잘못된 test_size 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")

        with pytest.raises(ValueError):
            dataset_service.train_test_split(test_size=1.5)

    def test_save_dataset_csv(self, dataset_service, sample_csv, tmp_path):
        """CSV 저장 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        filepath = tmp_path / "output.csv"

        result = dataset_service.save_dataset(str(filepath), format="csv")

        assert result["status"] == "success"
        assert Path(filepath).exists()

    def test_reset_data(self, dataset_service, sample_csv):
        """데이터 초기화 테스트"""
        dataset_service.load_dataset(sample_csv, "test.csv")
        original_len = len(dataset_service.data)

        # 데이터 수정
        dataset_service.data = dataset_service.data.head(2)
        assert len(dataset_service.data) == 2

        # 초기화
        result = dataset_service.reset_data()
        assert result["status"] == "success"
        assert len(dataset_service.data) == original_len


# ========================================
# API 엔드포인트 테스트
# ========================================


class TestDatasetAPI:
    """데이터셋 API 테스트"""

    def test_dataset_health(self):
        """헬스 체크"""
        response = client.get("/dataset/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_upload_csv(self, sample_csv):
        """CSV 파일 업로드"""
        response = client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "file_info" in response.json()

    def test_upload_json(self, sample_json):
        """JSON 파일 업로드"""
        response = client.post(
            "/dataset/upload",
            files={"file": ("test.json", io.BytesIO(sample_json), "application/json")},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_get_data_info_no_data(self):
        """데이터 정보 조회 - 데이터 없음"""
        response = client.get("/dataset/info")
        assert response.status_code == 400

    def test_get_data_info_with_data(self, sample_csv):
        """데이터 정보 조회 - 데이터 있음"""
        # 먼저 데이터 업로드
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.get("/dataset/info")
        assert response.status_code == 200
        data = response.json()
        assert "shape" in data
        assert "size_mb" in data

    def test_get_preview(self, sample_csv):
        """데이터 미리보기"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.get("/dataset/preview?n_rows=3")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_clean_data_missing_values(self, sample_csv_with_missing):
        """데이터 정제 - 결측치 처리"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv_with_missing), "text/csv")},
        )

        response = client.post(
            "/dataset/clean",
            json={"operation": "missing_values", "kwargs": {"strategy": "drop"}},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_clean_data_duplicates(self):
        """데이터 정제 - 중복 제거"""
        df = pd.DataFrame({"id": [1, 2, 2], "text": ["a", "b", "b"]})
        csv_data = df.to_csv(index=False).encode()

        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
        )

        response = client.post(
            "/dataset/clean",
            json={"operation": "duplicates", "kwargs": {}},
        )

        assert response.status_code == 200

    def test_analyze_tokens(self, sample_csv):
        """토큰 분석"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.post(
            "/dataset/analyze-tokens?column=text&model_name=gpt2"
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert "min_tokens" in data
        assert "max_tokens" in data

    def test_eda_statistics(self, sample_csv):
        """EDA - 통계"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.get("/dataset/eda/statistics")
        assert response.status_code == 200
        data = response.json()["data"]
        assert "total_rows" in data
        assert "total_columns" in data

    def test_eda_missing_values(self, sample_csv_with_missing):
        """EDA - 결측치 분석"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv_with_missing), "text/csv")},
        )

        response = client.get("/dataset/eda/missing-values")
        assert response.status_code == 200

    def test_eda_value_distribution(self, sample_csv):
        """EDA - 값 분포"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.get("/dataset/eda/value-distribution?column=label")
        assert response.status_code == 200

    def test_eda_summary(self, sample_csv):
        """EDA - 종합 요약"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.get("/dataset/eda/summary")
        assert response.status_code == 200
        summary = response.json()["summary"]
        assert "total_rows" in summary
        assert "columns" in summary

    def test_split_dataset(self, sample_csv):
        """데이터 분할"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.post(
            "/dataset/split",
            json={"test_size": 0.2, "random_state": 42},
        )

        assert response.status_code == 200
        data = response.json()
        assert "train_rows" in data
        assert "test_rows" in data

    def test_save_dataset(self, sample_csv, tmp_path):
        """데이터셋 저장"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        filepath = str(tmp_path / "output.csv")
        response = client.post(
            "/dataset/save",
            params={"filepath": filepath, "format": "csv"},
        )

        assert response.status_code == 200

    def test_reset_data(self, sample_csv):
        """데이터 초기화"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.post("/dataset/reset")
        assert response.status_code == 200


# ========================================
# 에러 처리 테스트
# ========================================


class TestErrorHandling:
    """에러 처리 테스트"""

    def test_upload_unsupported_format(self):
        """지원하지 않는 파일 형식"""
        response = client.post(
            "/dataset/upload",
            files={"file": ("test.txt", io.BytesIO(b"test"), "text/plain")},
        )

        assert response.status_code == 400

    def test_clean_invalid_operation(self, sample_csv):
        """잘못된 정제 작업"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.post(
            "/dataset/clean",
            json={"operation": "invalid_operation", "kwargs": {}},
        )

        assert response.status_code == 400

    def test_analyze_tokens_invalid_model(self, sample_csv):
        """토큰 분석 - 잘못된 모델"""
        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(sample_csv), "text/csv")},
        )

        response = client.post(
            "/dataset/analyze-tokens?column=text&model_name=nonexistent-model-xyz"
        )

        assert response.status_code == 400


# ========================================
# 성능 테스트
# ========================================


class TestPerformance:
    """성능 테스트"""

    def test_load_large_csv_performance(self):
        """대용량 CSV 로드 성능"""
        import time

        # 1000개 행 생성
        df = pd.DataFrame(
            {
                "id": range(1000),
                "text": ["sample text"] * 1000,
                "label": ["A"] * 1000,
            }
        )
        csv_data = df.to_csv(index=False).encode()

        start = time.time()
        response = client.post(
            "/dataset/upload",
            files={"file": ("large.csv", io.BytesIO(csv_data), "text/csv")},
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0  # 2초 이내

    def test_eda_performance(self):
        """EDA 성능"""
        import time

        df = pd.DataFrame(
            {
                "id": range(100),
                "text": ["sample"] * 100,
                "value": np.random.rand(100),
            }
        )
        csv_data = df.to_csv(index=False).encode()

        client.post(
            "/dataset/upload",
            files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
        )

        start = time.time()
        client.get("/dataset/eda/statistics")
        client.get("/dataset/eda/missing-values")
        client.get("/dataset/eda/summary")
        elapsed = time.time() - start

        assert elapsed < 1.0  # 1초 이내


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
