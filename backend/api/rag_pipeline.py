"""RAG (Retrieval-Augmented Generation) API"""
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.services.rag_service import RAGService

router = APIRouter(tags=["rag"])
rag_service = RAGService()

# ========================================
# 요청/응답 모델
# ========================================


class RAGConfigRequest(BaseModel):
    """RAG 설정 요청"""

    chunk_size: int = Field(512, description="청크 크기")
    chunk_overlap: int = Field(50, description="청크 겹침")
    top_k: int = Field(5, description="검색 결과 개수")


class SearchRequest(BaseModel):
    """검색 요청"""

    query: str = Field(..., description="검색 쿼리")
    top_k: Optional[int] = Field(None, description="검색 결과 개수")


class RAGChatRequest(BaseModel):
    """RAG 대화 요청"""

    message: str = Field(..., description="사용자 메시지")
    system_prompt: Optional[str] = Field(None, description="시스템 프롬프트")
    top_k: Optional[int] = Field(None, description="검색 결과 개수")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="온도")
    max_length: int = Field(512, ge=128, le=2048, description="최대 길이")


class LoadTextRequest(BaseModel):
    """텍스트 로드 요청"""
    
    text: str = Field(..., description="텍스트")


# ========================================
# 헬스 체크
# ========================================


@router.get("/health", tags=["Health"])
async def rag_health() -> Dict[str, str]:
    """RAG 서비스 헬스 체크"""
    return {"status": "ok", "service": "rag"}


# ========================================
# 초기화
# ========================================


@router.post("/initialize")
async def initialize_rag() -> Dict:
    """RAG 서비스 초기화"""
    try:
        result = rag_service.initialize()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"초기화 실패: {str(e)}")


# ========================================
# 문서 로드
# ========================================


@router.post("/load-pdf")
async def load_pdf(file: UploadFile = File(...)) -> Dict:
    """PDF 파일 로드"""
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise ValueError("PDF 파일만 지원합니다.")

        # 임시 파일로 저장
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # PDF 로드
        result = rag_service.load_pdf(tmp_path)

        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 로드 실패: {str(e)}")


@router.post("/load-text")
async def load_text(request: LoadTextRequest) -> Dict:
    """텍스트 로드"""
    try:
        result = rag_service.load_text(request.text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"텍스트 로드 실패: {str(e)}")


# ========================================
# 임베딩
# ========================================


@router.post("/create-embeddings")
async def create_embeddings() -> Dict:
    """임베딩 생성"""
    try:
        result = rag_service.create_embeddings()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"임베딩 생성 실패: {str(e)}")


# ========================================
# 검색
# ========================================


@router.post("/search")
async def search(request: SearchRequest) -> Dict:
    """유사도 검색"""
    try:
        result = rag_service.search(request.query, request.top_k)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"검색 실패: {str(e)}")


# ========================================
# RAG 대화
# ========================================


@router.post("/chat")
async def rag_chat(request: RAGChatRequest) -> Dict:
    """RAG 기반 대화"""
    try:
        result = rag_service.rag_chat(
            user_message=request.message,
            system_prompt=request.system_prompt,
            top_k=request.top_k,
            temperature=request.temperature,
            max_length=request.max_length,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"대화 실패: {str(e)}")


# ========================================
# 문서 관리
# ========================================


@router.get("/documents/info")
async def get_documents_info() -> Dict:
    """로드된 문서 정보"""
    try:
        result = rag_service.get_documents_info()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/documents/clear")
async def clear_documents() -> Dict:
    """문서 초기화"""
    try:
        result = rag_service.clear_documents()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# RAG 설정
# ========================================


@router.post("/config")
async def configure_rag(request: RAGConfigRequest) -> Dict:
    """RAG 설정"""
    try:
        result = rag_service.configure_rag(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            top_k=request.top_k,
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config")
async def get_rag_config() -> Dict:
    """RAG 설정 조회"""
    try:
        result = rag_service.get_rag_config()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 대화 히스토리
# ========================================


@router.get("/history")
async def get_conversation_history() -> Dict:
    """대화 히스토리 조회"""
    try:
        result = rag_service.get_conversation_history()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/history/clear")
async def clear_conversation_history() -> Dict:
    """대화 히스토리 초기화"""
    try:
        result = rag_service.clear_conversation_history()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# 통계
# ========================================


@router.get("/statistics")
async def get_statistics() -> Dict:
    """RAG 통계"""
    try:
        result = rag_service.get_statistics()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
