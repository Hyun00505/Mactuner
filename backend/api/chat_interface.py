"""채팅 인터페이스 API"""
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.model_loader import get_cached_model
from backend.services.chat_service import ChatService

router = APIRouter(tags=["chat"])
chat_service = ChatService()

# ========================================
# 요청/응답 모델
# ========================================


class InitializeChatRequest(BaseModel):
    """Chat 초기화 요청"""

    system_prompt: str = Field(
        "You are a helpful AI assistant. Answer questions clearly and concisely. "
        "Remember what the user told you. Be friendly and natural. Do not repeat yourself.",
        description="System prompt for the chat",
    )


class ChatRequest(BaseModel):
    """채팅 요청"""

    message: str = Field(..., description="User message")
    max_length: int = Field(512, ge=128, le=2048, description="Max response length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(50, ge=0, le=100, description="Top-k sampling")
    maintain_history: bool = Field(True, description="Maintain conversation history")
    max_tokens: int = Field(1024, ge=128, le=4096, description="Max tokens to generate")
    repeat_penalty: float = Field(1.1, ge=0.0, le=2.0, description="Repeat penalty")
    n_gpu_layers: int = Field(35, ge=0, le=100, description="GPU layers for GGUF models")


class GenerateRequest(BaseModel):
    """텍스트 생성 요청"""

    prompt: str = Field(..., description="Prompt for generation")
    max_length: int = Field(512, ge=128, le=2048, description="Max length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p")
    top_k: int = Field(50, ge=0, le=100, description="Top-k")


class SystemPromptRequest(BaseModel):
    """시스템 프롬프트 요청"""

    prompt: str = Field(..., description="System prompt")


class RecommendedParametersRequest(BaseModel):
    """파라미터 추천 요청"""

    response_style: str = Field(
        "balanced",
        description="Response style (creative, balanced, focused, deterministic)",
    )


# ========================================
# 헬스 체크
# ========================================


@router.get("/health", tags=["Health"])
async def chat_health() -> Dict[str, str]:
    """Chat 서비스 헬스 체크"""
    return {"status": "ok", "service": "chat"}


# ========================================
# Chat 초기화
# ========================================


@router.post("/initialize")
async def initialize_chat(request: InitializeChatRequest) -> Dict:
    """Chat 서비스 초기화"""
    try:
        # 모델 캐시에서 가져오기
        model_cache = get_cached_model()
        model = model_cache["model"]
        tokenizer = model_cache["tokenizer"]

        # Chat 서비스 초기화
        result = chat_service.initialize_from_model(
            model=model,
            tokenizer=tokenizer,
            system_prompt=request.system_prompt,
        )

        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat 초기화 실패: {str(e)}")


# ========================================
# 대화
# ========================================


@router.post("/chat")
async def chat(request: ChatRequest) -> Dict:
    """사용자 메시지에 응답"""
    try:
        # 모델이 초기화되지 않았으면 캐시에서 로드
        if chat_service.model is None or chat_service.tokenizer is None:
            try:
                model_cache = get_cached_model()
                is_gguf = model_cache.get("is_gguf", False)
                
                if is_gguf:
                    # GGUF 모델 (llama.cpp)
                    llama_service = model_cache["model"]
                    result = llama_service.chat(
                        user_message=request.message,
                        system_prompt=chat_service.system_prompt if hasattr(chat_service, 'system_prompt') else None,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        maintain_history=request.maintain_history,
                    )
                    return {"status": "success", "data": result}
                else:
                    # 일반 모델 (Transformers)
                    model = model_cache["model"]
                    tokenizer = model_cache["tokenizer"]
                    
                    # Chat 서비스 초기화
                    chat_service.initialize_from_model(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=chat_service.system_prompt,
                    )
            except Exception as init_error:
                raise HTTPException(
                    status_code=400, 
                    detail=f"모델 초기화 실패: {str(init_error)}"
                )
        
        # GGUF 모델인 경우
        if hasattr(chat_service, 'model') and str(type(chat_service.model)).find('LlamaCppService') != -1:
            result = chat_service.model.chat(
                user_message=request.message,
                system_prompt=chat_service.system_prompt if hasattr(chat_service, 'system_prompt') else None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                maintain_history=request.maintain_history,
                repeat_penalty=request.repeat_penalty,
                n_gpu_layers=request.n_gpu_layers,
            )
            return {"status": "success", "data": result}
        
        # 일반 모델인 경우
        result = chat_service.chat(
            user_message=request.message,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            maintain_history=request.maintain_history,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"채팅 실패: {str(e)}")


# ========================================
# 텍스트 생성 (히스토리 없이)
# ========================================


@router.post("/generate")
async def generate(request: GenerateRequest) -> Dict:
    """프롬프트로부터 텍스트 생성 (히스토리 미유지)"""
    try:
        result = chat_service.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"생성 실패: {str(e)}")


# ========================================
# 대화 히스토리
# ========================================


@router.get("/history")
async def get_history() -> Dict:
    """대화 히스토리 조회"""
    try:
        result = chat_service.get_conversation_history()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history/summary")
async def get_history_summary() -> Dict:
    """대화 히스토리 요약"""
    try:
        result = chat_service.get_history_summary()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/history/clear")
async def clear_history() -> Dict:
    """대화 히스토리 초기화"""
    try:
        result = chat_service.clear_history()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 시스템 프롬프트
# ========================================


@router.post("/system-prompt")
async def set_system_prompt(request: SystemPromptRequest) -> Dict:
    """시스템 프롬프트 설정"""
    try:
        result = chat_service.set_system_prompt(request.prompt)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/system-prompt")
async def get_system_prompt() -> Dict:
    """시스템 프롬프트 조회"""
    try:
        result = chat_service.get_system_prompt()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 파라미터 추천
# ========================================


@router.post("/recommended-parameters")
async def get_recommended_parameters(
    request: RecommendedParametersRequest,
) -> Dict:
    """응답 스타일에 따른 권장 파라미터"""
    try:
        result = chat_service.get_recommended_parameters(
            response_style=request.response_style
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 토큰 통계
# ========================================


@router.get("/token-statistics")
async def get_token_statistics() -> Dict:
    """토큰 사용 통계"""
    try:
        result = chat_service.get_token_statistics()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 상태
# ========================================


@router.get("/status")
async def get_chat_status() -> Dict:
    """Chat 서비스 상태"""
    try:
        result = chat_service.get_status()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
