"""채팅 서비스 테스트"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.chat_service import ChatService, Message

client = TestClient(app)


# ========================================
# 픽스처
# ========================================


@pytest.fixture
def chat_service():
    """Chat 서비스 인스턴스"""
    return ChatService()


# ========================================
# Message 클래스 테스트
# ========================================


class TestMessage:
    """Message 클래스 테스트"""

    def test_message_creation(self):
        """메시지 생성"""
        msg = Message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """메시지를 딕셔너리로 변환"""
        msg = Message("assistant", "Hi there")
        msg_dict = msg.to_dict()
        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Hi there"


# ========================================
# ChatService 단위 테스트
# ========================================


class TestChatService:
    """Chat 서비스 테스트"""

    def test_initialization(self, chat_service):
        """초기화 테스트"""
        assert chat_service.model is None
        assert chat_service.tokenizer is None
        assert len(chat_service.conversation_history) == 0
        assert chat_service.system_prompt == "You are a helpful AI assistant."

    def test_set_system_prompt(self, chat_service):
        """시스템 프롬프트 설정"""
        new_prompt = "You are a math expert."
        result = chat_service.set_system_prompt(new_prompt)

        assert result["status"] == "success"
        assert chat_service.system_prompt == new_prompt

    def test_get_system_prompt(self, chat_service):
        """시스템 프롬프트 조회"""
        chat_service.set_system_prompt("Custom prompt")
        result = chat_service.get_system_prompt()

        assert result["status"] == "success"
        assert result["system_prompt"] == "Custom prompt"

    def test_get_recommended_parameters_creative(self, chat_service):
        """파라미터 추천 - creative"""
        result = chat_service.get_recommended_parameters("creative")

        assert result["status"] == "success"
        assert result["style"] == "creative"
        assert result["parameters"]["temperature"] == 0.9

    def test_get_recommended_parameters_balanced(self, chat_service):
        """파라미터 추천 - balanced"""
        result = chat_service.get_recommended_parameters("balanced")

        assert result["parameters"]["temperature"] == 0.7
        assert result["parameters"]["num_beams"] == 1

    def test_get_recommended_parameters_focused(self, chat_service):
        """파라미터 추천 - focused"""
        result = chat_service.get_recommended_parameters("focused")

        assert result["parameters"]["temperature"] == 0.3
        assert result["parameters"]["max_length"] == 256

    def test_get_recommended_parameters_deterministic(self, chat_service):
        """파라미터 추천 - deterministic"""
        result = chat_service.get_recommended_parameters("deterministic")

        assert result["parameters"]["temperature"] == 0.0
        assert result["parameters"]["do_sample"] == False
        assert result["parameters"]["num_beams"] == 3

    def test_get_recommended_parameters_invalid_style(self, chat_service):
        """파라미터 추천 - 잘못된 스타일"""
        result = chat_service.get_recommended_parameters("invalid_style")

        # 기본값으로 balanced 반환
        assert result["style"] == "balanced"

    def test_get_conversation_history_empty(self, chat_service):
        """히스토리 조회 - 빈 상태"""
        result = chat_service.get_conversation_history()

        assert result["status"] == "success"
        assert result["history"] == []
        assert result["total_messages"] == 0

    def test_get_history_summary_empty(self, chat_service):
        """히스토리 요약 - 빈 상태"""
        result = chat_service.get_history_summary()

        assert result["status"] == "success"
        assert result["message_count"] == 0

    def test_clear_history(self, chat_service):
        """히스토리 초기화"""
        # 메시지 추가
        chat_service.conversation_history.append(Message("user", "Hello"))
        assert len(chat_service.conversation_history) == 1

        # 초기화
        result = chat_service.clear_history()
        assert result["status"] == "success"
        assert len(chat_service.conversation_history) == 0

    def test_get_token_statistics_empty(self, chat_service):
        """토큰 통계 - 빈 상태"""
        result = chat_service.get_token_statistics()

        assert result["status"] == "success"
        assert result["total_messages"] == 0
        assert result["total_tokens"] == 0

    def test_get_status(self, chat_service):
        """상태 조회"""
        result = chat_service.get_status()

        assert result["status"] == "success"
        assert result["initialized"] == False
        assert result["model_loaded"] == False
        assert result["conversation_length"] == 0

    def test_build_context_without_history(self, chat_service):
        """컨텍스트 구성 - 히스토리 없음"""
        context = chat_service._build_context(use_history=False)
        assert context == chat_service.system_prompt

    def test_recommended_parameters_all_styles(self, chat_service):
        """모든 스타일의 파라미터 확인"""
        styles = ["creative", "balanced", "focused", "deterministic"]

        for style in styles:
            result = chat_service.get_recommended_parameters(style)
            assert result["status"] == "success"
            assert result["style"] == style
            assert "parameters" in result
            assert "description" in result["parameters"]


# ========================================
# API 엔드포인트 테스트
# ========================================


class TestChatAPI:
    """Chat API 테스트"""

    def test_chat_health(self):
        """헬스 체크"""
        response = client.get("/chat/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_initialize_chat_no_model(self):
        """Chat 초기화 - 모델 없음"""
        response = client.post(
            "/chat/initialize",
            json={"system_prompt": "Custom prompt"},
        )

        assert response.status_code == 400

    def test_set_system_prompt(self):
        """시스템 프롬프트 설정"""
        response = client.post(
            "/chat/system-prompt",
            json={"prompt": "You are a helpful assistant."},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_get_system_prompt(self):
        """시스템 프롬프트 조회"""
        # 먼저 설정
        client.post(
            "/chat/system-prompt",
            json={"prompt": "Test prompt"},
        )

        response = client.get("/chat/system-prompt")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_recommended_parameters(self):
        """파라미터 추천"""
        response = client.post(
            "/chat/recommended-parameters",
            json={"response_style": "creative"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "parameters" in data["data"]

    def test_get_history(self):
        """히스토리 조회"""
        response = client.get("/chat/history")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_history_summary(self):
        """히스토리 요약"""
        response = client.get("/chat/history/summary")
        assert response.status_code == 200

    def test_clear_history(self):
        """히스토리 초기화"""
        response = client.post("/chat/history/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_token_statistics(self):
        """토큰 통계"""
        response = client.get("/chat/token-statistics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_chat_status(self):
        """Chat 상태"""
        response = client.get("/chat/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_chat_without_initialization(self):
        """채팅 - 초기화 없음"""
        response = client.post(
            "/chat/chat",
            json={"message": "Hello"},
        )

        assert response.status_code == 400

    def test_generate_without_initialization(self):
        """생성 - 초기화 없음"""
        response = client.post(
            "/chat/generate",
            json={"prompt": "Write a story"},
        )

        assert response.status_code == 400


# ========================================
# 파라미터 검증 테스트
# ========================================


class TestParameterValidation:
    """파라미터 검증 테스트"""

    def test_chat_request_invalid_temperature(self):
        """채팅 요청 - 잘못된 temperature"""
        response = client.post(
            "/chat/chat",
            json={
                "message": "Hello",
                "temperature": 3.0,  # 2.0 초과
            },
        )

        # Validation error
        assert response.status_code in [400, 422]

    def test_chat_request_invalid_max_length(self):
        """채팅 요청 - 잘못된 max_length"""
        response = client.post(
            "/chat/chat",
            json={
                "message": "Hello",
                "max_length": 100,  # 128 미만
            },
        )

        assert response.status_code in [400, 422]

    def test_generate_request_valid_parameters(self):
        """생성 요청 - 유효한 파라미터"""
        # 실제 모델 없이는 실패하지만, 파라미터 검증은 통과
        response = client.post(
            "/chat/generate",
            json={
                "prompt": "Hello",
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
            },
        )

        # 400 (모델 없음) 이지만 422 (파라미터 검증 실패)는 아님
        assert response.status_code != 422


# ========================================
# 에러 처리 테스트
# ========================================


class TestErrorHandling:
    """에러 처리 테스트"""

    def test_initialize_with_invalid_prompt(self):
        """초기화 - 빈 프롬프트"""
        response = client.post(
            "/chat/initialize",
            json={"system_prompt": ""},  # 빈 문자열 허용
        )

        # 빈 문자열도 유효한 프롬프트
        assert response.status_code in [200, 400]

    def test_set_system_prompt_empty(self):
        """시스템 프롬프트 - 빈 문자열"""
        response = client.post(
            "/chat/system-prompt",
            json={"prompt": ""},
        )

        # 빈 문자열도 유효
        assert response.status_code == 200

    def test_recommended_parameters_invalid_style(self):
        """파라미터 추천 - 잘못된 스타일"""
        response = client.post(
            "/chat/recommended-parameters",
            json={"response_style": "nonexistent_style"},
        )

        # 기본값으로 처리 (balanced)
        assert response.status_code == 200


# ========================================
# 통합 테스트
# ========================================


class TestIntegration:
    """통합 테스트"""

    def test_system_prompt_workflow(self):
        """시스템 프롬프트 워크플로우"""
        # 설정
        set_response = client.post(
            "/chat/system-prompt",
            json={"prompt": "You are a teacher."},
        )
        assert set_response.status_code == 200

        # 조회
        get_response = client.get("/chat/system-prompt")
        assert get_response.status_code == 200
        assert get_response.json()["data"]["system_prompt"] == "You are a teacher."

    def test_parameters_workflow(self):
        """파라미터 추천 워크플로우"""
        styles = ["creative", "balanced", "focused", "deterministic"]

        for style in styles:
            response = client.post(
                "/chat/recommended-parameters",
                json={"response_style": style},
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["style"] == style

    def test_history_workflow(self):
        """히스토리 관리 워크플로우"""
        # 초기 상태
        hist1 = client.get("/chat/history")
        assert len(hist1.json()["data"]["history"]) == 0

        # 요약 조회
        summary = client.get("/chat/history/summary")
        assert summary.status_code == 200

        # 초기화
        clear = client.post("/chat/history/clear")
        assert clear.status_code == 200

        # 다시 조회
        hist2 = client.get("/chat/history")
        assert len(hist2.json()["data"]["history"]) == 0


# ========================================
# 성능 테스트
# ========================================


class TestPerformance:
    """성능 테스트"""

    def test_set_system_prompt_performance(self):
        """시스템 프롬프트 설정 성능"""
        import time

        start = time.time()
        for _ in range(100):
            client.post(
                "/chat/system-prompt",
                json={"prompt": "Prompt"},
            )
        elapsed = time.time() - start

        # 100회가 1초 이내
        assert elapsed < 1.0

    def test_recommended_parameters_performance(self):
        """파라미터 추천 성능"""
        import time

        start = time.time()
        for _ in range(100):
            client.post(
                "/chat/recommended-parameters",
                json={"response_style": "balanced"},
            )
        elapsed = time.time() - start

        # 100회가 1초 이내
        assert elapsed < 1.0

    def test_history_operations_performance(self):
        """히스토리 작업 성능"""
        import time

        start = time.time()
        for _ in range(50):
            client.get("/chat/history")
            client.get("/chat/history/summary")
            client.post("/chat/history/clear")
        elapsed = time.time() - start

        # 150개 작업이 2초 이내
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
