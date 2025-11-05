"""llama.cpp 기반 GGUF 모델 서비스"""
import logging
from typing import Any, Dict, Optional
import os

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger(__name__)


class LlamaCppService:
    """llama.cpp을 사용한 GGUF 모델 서비스"""

    def __init__(self):
        self.llm: Optional[Llama] = None
        self.model_path: Optional[str] = None
        self.conversation_history: list = []
        self.system_prompt = "You are a helpful AI assistant."

    @staticmethod
    def is_gguf_model(model_path: str) -> bool:
        """GGUF 모델인지 확인"""
        if not isinstance(model_path, str):
            return False
        return model_path.lower().endswith('.gguf')

    @staticmethod
    def is_available() -> bool:
        """llama-cpp-python이 설치되어 있는지 확인"""
        return LLAMA_CPP_AVAILABLE

    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int = 35,
        n_threads: int = 4,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """GGUF 모델 로드"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                return {
                    "status": "error",
                    "message": "llama-cpp-python not installed. Install with: pip install llama-cpp-python",
                }

            if not os.path.exists(model_path):
                return {
                    "status": "error",
                    "message": f"Model file not found: {model_path}",
                }

            logger.info(f"Loading GGUF model with llama.cpp: {model_path}")

            # llama.cpp 모델 로드
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,  # Metal (Mac GPU) 레이어 수
                n_threads=n_threads,
                n_batch=n_batch,
                n_ctx=2048,  # Context window 크기를 2048로 증가
                verbose=verbose,
            )

            self.model_path = model_path
            self.conversation_history = []

            return {
                "status": "success",
                "message": f"GGUF model loaded successfully with llama.cpp",
                "model_path": model_path,
                "model_type": "GGUF (llama.cpp)",
                "device": "Metal (Mac GPU) + CPU",
            }

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to load GGUF model: {str(e)}",
            }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """텍스트 생성"""
        try:
            if self.llm is None:
                return {
                    "status": "error",
                    "message": "Model not loaded",
                }

            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False,
            )

            text = response["choices"][0]["text"].strip()

            return {
                "status": "success",
                "response": text,
                "tokens_used": response.get("usage", {}).get("completion_tokens", 0),
            }

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Generation failed: {str(e)}",
            }

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        maintain_history: bool = True,
    ) -> Dict[str, Any]:
        """채팅 응답"""
        try:
            if self.llm is None:
                return {
                    "status": "error",
                    "message": "Model not loaded",
                }

            if system_prompt:
                self.system_prompt = system_prompt

            # 컨텍스트 구축
            if maintain_history:
                context = self._build_context_with_history(user_message)
            else:
                context = self._build_context(user_message)

            # 응답 생성
            response = self.llm(
                context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False,
                stop=["User:", "Assistant:", "\n\n"],
            )

            text = response["choices"][0]["text"].strip()

            # 히스토리 유지
            if maintain_history:
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append(
                    {"role": "assistant", "content": text}
                )

                # 최근 10개 메시지만 유지
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]

            return {
                "status": "success",
                "response": text,
                "tokens_used": response.get("usage", {}).get("completion_tokens", 0),
            }

        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Chat failed: {str(e)}",
            }

    def _build_context(self, user_message: str) -> str:
        """컨텍스트 구축 (히스토리 없이)"""
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"User: {user_message}\n"
        prompt += "Assistant:"
        return prompt

    def _build_context_with_history(self, user_message: str) -> str:
        """컨텍스트 구축 (히스토리 포함)"""
        prompt = f"{self.system_prompt}\n\n"

        # 최근 대화 추가
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"

        prompt += f"User: {user_message}\n"
        prompt += "Assistant:"
        return prompt

    def clear_history(self) -> Dict[str, str]:
        """히스토리 초기화"""
        self.conversation_history = []
        return {"status": "success", "message": "History cleared"}

    def get_history(self) -> Dict[str, Any]:
        """히스토리 조회"""
        return {
            "status": "success",
            "history": self.conversation_history,
            "count": len(self.conversation_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        return {
            "status": "success",
            "model_loaded": self.llm is not None,
            "model_path": self.model_path,
            "model_type": "GGUF (llama.cpp)",
            "device": "Metal (Mac GPU) + CPU",
            "history_size": len(self.conversation_history),
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
        }
