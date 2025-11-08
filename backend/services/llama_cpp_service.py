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
        self.system_prompt = ""  # Llama 모델용 빈 시스템 프롬프트

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
        repeat_penalty: float = 1.1,
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
                repeat_penalty=repeat_penalty,
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
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        maintain_history: bool = True,
        repeat_penalty: float = 1.1,
        n_gpu_layers: Optional[int] = None,
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

            # max_tokens이 너무 작으면 조정 (더 자연스러운 응답을 위해)
            # 프론트엔드의 auto mode에서는 이미 계산된 값이 전달되지만,
            # 백엔드에서도 한 번 더 검증
            effective_max_tokens = max(max_tokens, 256)
            if effective_max_tokens < 512:
                effective_max_tokens = 512  # 최소 512 토큰으로 조정

            # 응답 생성 - max_tokens를 늘려서 완전한 응답을 얻도록
            try:
                response = self.llm(
                    context,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    echo=False,
                    stop=["Q:", "\n\nQ:"],  # 새로운 질문이 시작되면 중단
                )
            except Exception as e:
                # Context window 초과 시 더 짧은 히스토리로 재시도
                if "exceed context window" in str(e):
                    logger.warning(f"Context window exceeded, retrying with shorter history: {e}")
                    self.conversation_history = self.conversation_history[-6:]  # 최근 3개 왕복만
                    context = self._build_context_with_history(user_message)
                    response = self.llm(
                        context,
                        max_tokens=effective_max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        echo=False,
                        stop=["Q:", "\n\nQ:"],
                    )
                else:
                    raise

            text = response["choices"][0]["text"].strip()
            
            # 응답 검증 - 응답이 비어있거나 유효하지 않으면 재시도
            if not text or len(text.strip()) < 5:
                logger.warning(f"Empty or very short response: '{text}'. Retrying...")
                # 재시도: 더 낮은 온도로 다시 시도
                try:
                    # 재시도: 기본값으로 설정하되 최소 512 토큰 유지
                    retry_max_tokens = max(effective_max_tokens, 512)
                    response = self.llm(
                        context,
                        max_tokens=retry_max_tokens,
                        temperature=max(0.3, temperature * 0.5),  # 온도 감소
                        top_p=0.8,
                        top_k=40,
                        repeat_penalty=1.2,
                        echo=False,
                        stop=["Q:", "\n\nQ:"],
                    )
                    text = response["choices"][0]["text"].strip()
                except Exception as retry_error:
                    logger.error(f"Retry failed: {str(retry_error)}")
                    text = "I'm processing your question. Could you please provide more context or rephrase it?"
            
            # 응답 정리 - 불완전한 문장 제거
            text = self._cleanup_response(text)

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
    
    def _cleanup_response(self, text: str) -> str:
        """응답 정리 - 불완전한 코드나 문장 제거"""
        if not text or not text.strip():
            return "I'm unable to generate a response. Please try again with different parameters or a simpler prompt."
        
        # 기본 정리
        text = text.strip()
        
        # 줄 단위로 분할
        lines = text.split('\n')
        
        # 선행/후행 공백 제거, 빈 줄은 유지 (마크다운 포맷을 위해)
        lines = [line.rstrip() for line in lines]
        
        # 프롬프트 구조 제거 (Q:, A: 등)
        cleaned_lines = []
        for line in lines:
            # 프롬프트 마커로 시작하면 건너뛰기
            if line.strip().startswith(('Q:', 'A:', 'User:', 'Assistant:', '###')):
                continue
            cleaned_lines.append(line)
        
        # 빈 줄 제거 (결과 검증 전에)
        non_empty_lines = [line for line in cleaned_lines if line.strip()]
        
        # 응답이 너무 짧으면 원본 반환 (정리 전 텍스트가 있다면)
        if len('\n'.join(non_empty_lines).strip()) < 5 and len(non_empty_lines) == 0:
            return text  # 원본 텍스트 반환
        
        # 코드 블록 완성도 확인
        code_fence_count = sum(line.count('```') for line in cleaned_lines)
        
        # 코드 블록이 홀수개면 닫기 추가
        if code_fence_count % 2 == 1:
            cleaned_lines.append('```')
        
        result = '\n'.join(cleaned_lines).strip()
        return result if result else text  # 비어있으면 원본 반환

    def _build_context(self, user_message: str) -> str:
        """컨텍스트 구축 (히스토리 없이)"""
        # 시스템 프롬프트가 있으면 포함
        prompt = ""
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n"
        
        # 사용자 질문 추가
        prompt += f"Q: {user_message}\n\nA:"
        return prompt.strip()

    def _build_context_with_history(self, user_message: str) -> str:
        """컨텍스트 구축 (히스토리 포함)"""
        prompt = ""
        
        # 시스템 프롬프트가 있으면 포함
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n"
        
        # 최근 대화 추가 (최대 5개 왕복)
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        # 대화 히스토리 추가
        for msg in recent_history:
            if msg["role"] == "user":
                prompt += f"Q: {msg['content']}\n"
            else:
                prompt += f"A: {msg['content']}\n\n"
        
        # 새로운 질문 추가
        prompt += f"Q: {user_message}\n\nA:"
        return prompt.strip()

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
