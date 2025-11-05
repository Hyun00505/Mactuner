"""채팅 서비스"""
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.utils.mac_optimization import MACOptimizer

logger = logging.getLogger(__name__)


class Message:
    """대화 메시지"""

    def __init__(self, role: str, content: str):
        self.role = role  # "user" or "assistant"
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatService:
    """채팅 서비스"""

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.conversation_history: List[Message] = []
        self.device = MACOptimizer.get_device()
        self.system_prompt = "You are a helpful AI assistant."

    # ========================================
    # 초기화
    # ========================================

    def initialize_from_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> Dict[str, Any]:
        """모델과 토크나이저로 초기화"""
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.system_prompt = system_prompt
            self.conversation_history = []

            return {
                "status": "success",
                "message": "Chat service initialized",
                "system_prompt": system_prompt,
                "device": str(self.device),
            }

        except Exception as e:
            raise RuntimeError(f"Chat 서비스 초기화 실패: {str(e)}")

    # ========================================
    # 텍스트 생성
    # ========================================

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """텍스트 생성"""
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("모델이 초기화되지 않았습니다.")

            # 입력 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)

            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 결과 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # 프롬프트 부분 제거
            response_text = generated_text[len(prompt) :].strip()
            
            # 응답 정리
            response_text = self._cleanup_response(response_text)

            return {
                "status": "success",
                "response": response_text,
                "full_text": generated_text,
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": outputs.shape[1],
            }

        except Exception as e:
            raise RuntimeError(f"텍스트 생성 실패: {str(e)}")

    # ========================================
    # 대화 관리
    # ========================================

    def chat(
        self,
        user_message: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        maintain_history: bool = True,
    ) -> Dict[str, Any]:
        """사용자 메시지에 응답"""
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("모델이 초기화되지 않았습니다.")

            # 사용자 메시지 저장
            if maintain_history:
                self.conversation_history.append(Message("user", user_message))

            # 컨텍스트 구성
            context = self._build_context(maintain_history)

            # 응답 생성
            result = self.generate(
                prompt=context,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            response_text = result["response"]

            # 응답 저장
            if maintain_history:
                self.conversation_history.append(
                    Message("assistant", response_text)
                )

            return {
                "status": "success",
                "user_message": user_message,
                "response": response_text,
                "history_maintained": maintain_history,
                "history_length": len(self.conversation_history),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }

        except Exception as e:
            raise RuntimeError(f"채팅 실패: {str(e)}")

    def _build_context(self, use_history: bool = True) -> str:
        """대화 컨텍스트 구성"""
        if not use_history or len(self.conversation_history) == 0:
            return self.system_prompt

        # 명확한 프롬프트 형식
        context = f"{self.system_prompt}\n\n"
        context += "### Conversation History\n"
        
        # 최근 10개 메시지까지만 포함 (더 많은 컨텍스트)
        recent_history = self.conversation_history[-10:]
        
        for i, message in enumerate(recent_history):
            if message.role == "user":
                context += f"User: {message.content}\n"
            else:
                context += f"Assistant: {message.content}\n"
        
        # 명확한 구분자로 새로운 응답 요청
        context += "\n### Current Response\nAssistant:"
        return context

    def _cleanup_response(self, response: str) -> str:
        """응답 정리 - 반복 제거 및 포맷팅"""
        if not response:
            return response
        
        # 1. 기본 정리
        response = response.strip()
        
        # 2. "User:" 또는 "Assistant:" 포함된 부분은 제거 (프롬프트 누출 방지)
        lines = response.split('\n')
        result_lines = []
        for line in lines:
            line = line.strip()
            # 프롬프트 구조 제거
            if line.startswith(('User:', 'Assistant:', 'A:', 'U:', '###', 'Assistant')):
                continue
            if not line:
                continue
            result_lines.append(line)
        
        # 3. 반복 제거 (더 정교한 감지)
        final_lines = []
        for line in result_lines:
            # 이미 있는 문장과 정확히 같은지 확인
            if line in final_lines:
                continue
            
            # 유사한 문장 제거 (단어 유사도)
            is_repetitive = False
            if len(final_lines) >= 1:
                # 마지막 문장과 비교
                last_line = final_lines[-1]
                # 50% 이상 단어가 겹치면 반복으로 판단
                last_words = set(last_line.lower().split())
                current_words = set(line.lower().split())
                if len(last_words) > 0 and len(current_words) > 0:
                    overlap = len(last_words & current_words) / max(len(last_words), len(current_words))
                    if overlap > 0.5:
                        is_repetitive = True
            
            if not is_repetitive:
                final_lines.append(line)
        
        # 4. 첫 2줄만 사용 (더 짧게)
        if len(final_lines) > 2:
            final_lines = final_lines[:2]
        
        # 5. 결과 생성
        result = ' '.join(final_lines)
        
        # 6. 한글 문장 길이 제한 (너무 길면 끝내기)
        if len(result) > 200:  # 약 100자 한글
            result = result[:200] + '...'
        
        # 7. 마침표 추가
        if result and not result.endswith(('。', '。', '.', '!', '?', '?', '!', '...', '…')):
            # 한글인 경우 마침표 추가
            if any('\uac00' <= char <= '\ud7a3' for char in result):
                if not result.endswith(('다', '요', '가', '네')):
                    result += '다'
            else:
                result += '.'
        
        return result

    # ========================================
    # 대화 히스토리
    # ========================================

    def get_conversation_history(self) -> Dict[str, Any]:
        """대화 히스토리 조회"""
        return {
            "status": "success",
            "history": [msg.to_dict() for msg in self.conversation_history],
            "total_messages": len(self.conversation_history),
        }

    def clear_history(self) -> Dict[str, str]:
        """대화 히스토리 초기화"""
        self.conversation_history = []
        return {
            "status": "success",
            "message": "Conversation history cleared",
        }

    def get_history_summary(self) -> Dict[str, Any]:
        """대화 요약"""
        if len(self.conversation_history) == 0:
            return {
                "status": "success",
                "summary": "No conversation",
                "message_count": 0,
                "user_messages": 0,
                "assistant_messages": 0,
            }

        user_count = sum(1 for msg in self.conversation_history if msg.role == "user")
        assistant_count = (
            len(self.conversation_history) - user_count
        )

        # 첫 번째와 마지막 메시지
        first_msg = self.conversation_history[0].content[:100]
        last_msg = self.conversation_history[-1].content[:100]

        return {
            "status": "success",
            "total_messages": len(self.conversation_history),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "first_message": first_msg,
            "last_message": last_msg,
        }

    # ========================================
    # 시스템 프롬프트
    # ========================================

    def set_system_prompt(self, prompt: str) -> Dict[str, str]:
        """시스템 프롬프트 설정"""
        self.system_prompt = prompt
        return {
            "status": "success",
            "system_prompt": prompt,
        }

    def get_system_prompt(self) -> Dict[str, str]:
        """시스템 프롬프트 조회"""
        return {
            "status": "success",
            "system_prompt": self.system_prompt,
        }

    # ========================================
    # 파라미터 제시
    # ========================================

    def get_recommended_parameters(
        self,
        response_style: str = "balanced",
    ) -> Dict[str, Any]:
        """응답 스타일에 따른 권장 파라미터"""
        recommendations = {
            "creative": {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "max_length": 512,
                "num_beams": 1,
                "do_sample": True,
                "description": "창의적이고 다양한 응답",
            },
            "balanced": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "max_length": 512,
                "num_beams": 1,
                "do_sample": True,
                "description": "균형잡힌 응답",
            },
            "focused": {
                "temperature": 0.3,
                "top_p": 0.7,
                "top_k": 30,
                "repetition_penalty": 1.5,
                "max_length": 256,
                "num_beams": 1,
                "do_sample": True,
                "description": "집중된 짧은 응답",
            },
            "deterministic": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "max_length": 256,
                "num_beams": 3,
                "do_sample": False,
                "description": "일관성있는 고정된 응답",
            },
        }

        if response_style not in recommendations:
            response_style = "balanced"

        return {
            "status": "success",
            "style": response_style,
            "parameters": recommendations[response_style],
            "available_styles": list(recommendations.keys()),
        }

    # ========================================
    # 토큰 통계
    # ========================================

    def get_token_statistics(self) -> Dict[str, Any]:
        """토큰 통계"""
        if len(self.conversation_history) == 0:
            return {
                "status": "success",
                "total_messages": 0,
                "total_tokens": 0,
                "average_tokens_per_message": 0,
            }

        total_tokens = 0
        for msg in self.conversation_history:
            tokens = len(self.tokenizer.encode(msg.content))
            total_tokens += tokens

        avg_tokens = total_tokens / len(self.conversation_history)

        return {
            "status": "success",
            "total_messages": len(self.conversation_history),
            "total_tokens": total_tokens,
            "average_tokens_per_message": round(avg_tokens, 2),
        }

    # ========================================
    # 상태 정보
    # ========================================

    def get_status(self) -> Dict[str, Any]:
        """현재 상태"""
        return {
            "status": "success",
            "initialized": self.model is not None and self.tokenizer is not None,
            "device": str(self.device),
            "system_prompt": self.system_prompt,
            "conversation_length": len(self.conversation_history),
            "model_loaded": self.model is not None,
        }
