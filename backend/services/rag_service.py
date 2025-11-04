"""RAG (Retrieval-Augmented Generation) 서비스"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.config import settings

logger = logging.getLogger(__name__)


class DocumentChunk:
    """문서 청크"""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}
        self.embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }


class RAGService:
    """RAG 서비스"""

    def __init__(self):
        self.documents: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.chunk_size = settings.RAG_CHUNK_SIZE
        self.chunk_overlap = settings.RAG_CHUNK_OVERLAP
        self.top_k = settings.RAG_TOP_K

    # ========================================
    # 초기화
    # ========================================

    def initialize(self) -> Dict[str, Any]:
        """RAG 서비스 초기화 (임베딩 모델 로드)"""
        try:
            logger.info("RAG 서비스 초기화 중...")

            # 임베딩 모델 로드
            model_name = settings.RAG_EMBEDDING_MODEL
            logger.info(f"임베딩 모델 로드: {model_name}")

            self.embedding_model = SentenceTransformer(model_name)

            return {
                "status": "success",
                "message": "RAG 서비스 초기화 완료",
                "embedding_model": model_name,
                "device": str(next(self.embedding_model.parameters()).device),
            }

        except Exception as e:
            raise RuntimeError(f"RAG 초기화 실패: {str(e)}")

    # ========================================
    # 문서 처리
    # ========================================

    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF 파일 로드 및 처리"""
        try:
            file_path = Path(file_path).expanduser()

            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")

            try:
                import PyPDF2

                logger.info(f"PDF 로드: {file_path}")

                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    total_pages = len(pdf_reader.pages)
                    all_text = ""

                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        all_text += f"\n--- 페이지 {page_num + 1} ---\n{text}"

                # 텍스트 청크 생성
                chunks = self._create_chunks(all_text, file_path.name)

                self.documents = chunks

                return {
                    "status": "success",
                    "file_path": str(file_path),
                    "file_size_mb": file_path.stat().st_size / (1024**2),
                    "total_pages": total_pages,
                    "total_text_size_kb": len(all_text) / 1024,
                    "chunks_created": len(chunks),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }

            except ImportError:
                raise RuntimeError(
                    "PyPDF2가 설치되지 않았습니다. 설치: pip install PyPDF2"
                )

        except Exception as e:
            raise RuntimeError(f"PDF 로드 실패: {str(e)}")

    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """텍스트 로드 및 청크 생성"""
        try:
            logger.info("텍스트 로드 및 청크 생성 중...")

            # 텍스트 청크 생성
            chunks = self._create_chunks(text, "text_input")

            self.documents = chunks

            return {
                "status": "success",
                "total_text_size_kb": len(text) / 1024,
                "chunks_created": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

        except Exception as e:
            raise RuntimeError(f"텍스트 로드 실패: {str(e)}")

    def _create_chunks(
        self,
        text: str,
        source: str,
    ) -> List[DocumentChunk]:
        """텍스트를 청크로 분할"""
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]

            if len(chunk_text.strip()) > 0:
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": source,
                        "start_index": i,
                        "end_index": min(i + self.chunk_size, len(text)),
                    },
                )
                chunks.append(chunk)

        logger.info(f"총 {len(chunks)}개 청크 생성 완료")
        return chunks

    # ========================================
    # 임베딩
    # ========================================

    def create_embeddings(self) -> Dict[str, Any]:
        """문서 청크의 임베딩 생성"""
        try:
            if not self.documents:
                raise ValueError("처리할 문서가 없습니다.")

            if self.embedding_model is None:
                raise ValueError("임베딩 모델이 로드되지 않았습니다. initialize()를 먼저 호출하세요.")

            logger.info(f"총 {len(self.documents)}개 청크의 임베딩 생성 중...")

            # 모든 청크의 텍스트 추출
            texts = [chunk.content for chunk in self.documents]

            # 임베딩 생성
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            # 청크에 임베딩 저장
            for i, chunk in enumerate(self.documents):
                chunk.embedding = embeddings[i]

            self.embeddings = embeddings

            logger.info("임베딩 생성 완료")

            return {
                "status": "success",
                "total_chunks": len(self.documents),
                "embedding_dimension": embeddings.shape[1],
                "embedding_model": settings.RAG_EMBEDDING_MODEL,
            }

        except Exception as e:
            raise RuntimeError(f"임베딩 생성 실패: {str(e)}")

    # ========================================
    # 유사도 검색
    # ========================================

    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """유사도 검색"""
        try:
            if not self.documents or self.embeddings is None:
                raise ValueError("검색할 문서가 없습니다. 먼저 문서를 로드하고 임베딩을 생성하세요.")

            if self.embedding_model is None:
                raise ValueError("임베딩 모델이 로드되지 않았습니다.")

            if top_k is None:
                top_k = self.top_k

            logger.info(f"검색: '{query}' (top_k={top_k})")

            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

            # 코사인 유사도 계산
            similarities = self._calculate_similarities(query_embedding, self.embeddings)

            # 상위 k개 문서 선택
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                results.append(
                    {
                        "rank": len(results) + 1,
                        "document": self.documents[idx].to_dict(),
                        "similarity": float(similarities[idx]),
                    }
                )

            return {
                "status": "success",
                "query": query,
                "total_documents": len(self.documents),
                "results": results,
            }

        except Exception as e:
            raise RuntimeError(f"검색 실패: {str(e)}")

    def _calculate_similarities(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """코사인 유사도 계산"""
        # 벡터 정규화
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 코사인 유사도
        similarities = np.dot(embeddings_normalized, query_normalized)
        return similarities

    # ========================================
    # RAG 대화
    # ========================================

    def rag_chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """RAG 기반 대화"""
        try:
            # 1. 관련 문서 검색
            search_results = self.search(user_message, top_k)
            retrieved_docs = [r["document"] for r in search_results["results"]]

            # 2. 컨텍스트 구성
            context = self._build_context(retrieved_docs, system_prompt)

            # 3. 대화 히스토리에 저장
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )

            # 4. 응답 준비 정보 반환
            return {
                "status": "success",
                "user_message": user_message,
                "retrieved_documents": len(retrieved_docs),
                "top_documents": retrieved_docs[:3],
                "context": context,
                "temperature": temperature,
                "max_length": max_length,
            }

        except Exception as e:
            raise RuntimeError(f"RAG 대화 실패: {str(e)}")

    def _build_context(
        self,
        documents: List[Dict[str, Any]],
        system_prompt: Optional[str],
    ) -> str:
        """컨텍스트 구성"""
        context = ""

        if system_prompt:
            context += f"{system_prompt}\n\n"

        context += "## 관련 문서 정보\n"
        for i, doc in enumerate(documents, 1):
            context += f"\n### 문서 {i} (출처: {doc['metadata'].get('source', 'unknown')})\n"
            context += doc["content"][:500] + "...\n"

        context += "\n---\n\n## 질문에 대해 위 문서를 기반으로 답변하세요.\n"

        return context

    # ========================================
    # 문서 관리
    # ========================================

    def get_documents_info(self) -> Dict[str, Any]:
        """로드된 문서 정보"""
        if not self.documents:
            return {
                "status": "success",
                "total_documents": 0,
                "message": "로드된 문서가 없습니다.",
            }

        total_size = sum(len(doc.content) for doc in self.documents)

        return {
            "status": "success",
            "total_documents": len(self.documents),
            "total_text_size_kb": total_size / 1024,
            "embedding_created": self.embeddings is not None,
            "embedding_model": settings.RAG_EMBEDDING_MODEL if self.embedding_model else None,
        }

    def clear_documents(self) -> Dict[str, str]:
        """문서 초기화"""
        self.documents = []
        self.embeddings = None
        self.conversation_history = []

        return {
            "status": "success",
            "message": "문서가 초기화되었습니다.",
        }

    # ========================================
    # 설정 관리
    # ========================================

    def configure_rag(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """RAG 설정"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        return {
            "status": "success",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
        }

    def get_rag_config(self) -> Dict[str, Any]:
        """현재 RAG 설정 조회"""
        return {
            "status": "success",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embedding_model": settings.RAG_EMBEDDING_MODEL,
        }

    # ========================================
    # 대화 히스토리
    # ========================================

    def get_conversation_history(self) -> Dict[str, Any]:
        """대화 히스토리 조회"""
        return {
            "status": "success",
            "history": self.conversation_history,
            "total_messages": len(self.conversation_history),
        }

    def clear_conversation_history(self) -> Dict[str, str]:
        """대화 히스토리 초기화"""
        self.conversation_history = []

        return {
            "status": "success",
            "message": "대화 히스토리가 초기화되었습니다.",
        }

    # ========================================
    # 통계
    # ========================================

    def get_statistics(self) -> Dict[str, Any]:
        """RAG 통계"""
        if not self.documents:
            return {
                "status": "success",
                "total_documents": 0,
                "message": "로드된 문서가 없습니다.",
            }

        total_size = sum(len(doc.content) for doc in self.documents)
        avg_size = total_size / len(self.documents) if self.documents else 0

        return {
            "status": "success",
            "total_documents": len(self.documents),
            "total_text_size_kb": round(total_size / 1024, 2),
            "average_chunk_size_bytes": round(avg_size, 2),
            "embedding_created": self.embeddings is not None,
            "conversation_messages": len(self.conversation_history),
            "configuration": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
            },
        }
