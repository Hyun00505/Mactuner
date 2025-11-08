"""워크플로우 관리 API"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

# workflows 폴더 경로
WORKFLOWS_DIR = Path(__file__).parent.parent.parent / "workflows"
WORKFLOWS_DIR.mkdir(exist_ok=True)


class WorkflowRequest(BaseModel):
    """워크플로우 저장 요청"""
    workflow: Dict[str, Any]


class WorkflowResponse(BaseModel):
    """워크플로우 응답"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# ========================================
# 워크플로우 목록 조회
# ========================================


@router.get("/list")
async def list_workflows() -> Dict[str, Any]:
    """저장된 워크플로우 목록 조회"""
    try:
        workflows = []
        
        # workflows 폴더의 모든 JSON 파일 읽기
        for file_path in WORKFLOWS_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    workflow_data = json.load(f)
                    workflows.append({
                        "id": workflow_data.get("id", file_path.stem),
                        "name": workflow_data.get("name", file_path.stem),
                        "description": workflow_data.get("description", ""),
                        "createdAt": workflow_data.get("createdAt", 0),
                        "updatedAt": workflow_data.get("updatedAt", 0),
                        "version": workflow_data.get("version", "1.0.0"),
                        "filename": file_path.name,
                    })
            except Exception as e:
                print(f"⚠️ 워크플로우 파일 읽기 오류 ({file_path.name}): {str(e)}")
                continue
        
        # 생성일 기준 정렬 (최신순)
        workflows.sort(key=lambda x: x.get("createdAt", 0), reverse=True)
        
        return {
            "status": "success",
            "workflows": workflows,
            "count": len(workflows),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"워크플로우 목록 조회 실패: {str(e)}")


# ========================================
# 워크플로우 저장
# ========================================


@router.post("/save")
async def save_workflow(request: WorkflowRequest) -> Dict[str, Any]:
    """워크플로우를 JSON 파일로 저장"""
    try:
        workflow = request.workflow
        
        # 필수 필드 검증
        if not workflow.get("id"):
            raise ValueError("워크플로우 ID가 필요합니다")
        if not workflow.get("name"):
            raise ValueError("워크플로우 이름이 필요합니다")
        
        # 파일명 생성 (안전한 파일명)
        workflow_id = workflow.get("id", "workflow")
        safe_filename = "".join(c for c in workflow_id if c.isalnum() or c in ("-", "_"))
        filename = f"{safe_filename}.json"
        file_path = WORKFLOWS_DIR / filename
        
        # 워크플로우 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": f"워크플로우가 저장되었습니다: {filename}",
            "filename": filename,
            "path": str(file_path),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"워크플로우 저장 실패: {str(e)}")


# ========================================
# 워크플로우 불러오기
# ========================================


@router.get("/load/{filename}")
async def load_workflow(filename: str) -> Dict[str, Any]:
    """워크플로우 파일 불러오기"""
    try:
        # 파일명 검증 (보안)
        if not filename.endswith(".json"):
            filename += ".json"
        
        # 경로 검증 (상위 디렉토리 접근 방지)
        file_path = WORKFLOWS_DIR / filename
        if str(file_path.resolve()).startswith(str(WORKFLOWS_DIR.resolve())):
            # 경로가 유효함
            pass
        else:
            raise ValueError("잘못된 파일 경로입니다")
        
        if not file_path.exists():
            raise ValueError(f"워크플로우 파일을 찾을 수 없습니다: {filename}")
        
        # 워크플로우 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        
        return {
            "status": "success",
            "workflow": workflow,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"워크플로우 불러오기 실패: {str(e)}")


# ========================================
# 워크플로우 삭제
# ========================================


@router.delete("/delete/{filename}")
async def delete_workflow(filename: str) -> Dict[str, Any]:
    """워크플로우 파일 삭제"""
    try:
        # 파일명 검증
        if not filename.endswith(".json"):
            filename += ".json"
        
        # 경로 검증
        file_path = WORKFLOWS_DIR / filename
        if str(file_path.resolve()).startswith(str(WORKFLOWS_DIR.resolve())):
            # 경로가 유효함
            pass
        else:
            raise ValueError("잘못된 파일 경로입니다")
        
        if not file_path.exists():
            raise ValueError(f"워크플로우 파일을 찾을 수 없습니다: {filename}")
        
        # 파일 삭제
        file_path.unlink()
        
        return {
            "status": "success",
            "message": f"워크플로우가 삭제되었습니다: {filename}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"워크플로우 삭제 실패: {str(e)}")


# ========================================
# 워크플로우 업로드 (JSON 파일)
# ========================================


@router.post("/upload")
async def upload_workflow(file: UploadFile = File(...)) -> Dict[str, Any]:
    """JSON 파일로 워크플로우 업로드"""
    try:
        # 파일 확장자 검증
        if not file.filename.endswith(".json"):
            raise ValueError("JSON 파일만 업로드할 수 있습니다")
        
        # 파일 읽기
        content = await file.read()
        workflow = json.loads(content.decode("utf-8"))
        
        # 필수 필드 검증
        if not workflow.get("id"):
            raise ValueError("워크플로우 ID가 필요합니다")
        if not workflow.get("name"):
            raise ValueError("워크플로우 이름이 필요합니다")
        
        # 파일명 생성
        workflow_id = workflow.get("id", "workflow")
        safe_filename = "".join(c for c in workflow_id if c.isalnum() or c in ("-", "_"))
        filename = f"{safe_filename}.json"
        file_path = WORKFLOWS_DIR / filename
        
        # 워크플로우 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": f"워크플로우가 업로드되었습니다: {filename}",
            "filename": filename,
            "workflow": workflow,
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효하지 않은 JSON 파일입니다")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"워크플로우 업로드 실패: {str(e)}")


# ========================================
# 헬스 체크
# ========================================


@router.get("/health")
async def workflow_health() -> Dict[str, str]:
    """워크플로우 API 헬스 체크"""
    return {
        "status": "ok",
        "service": "workflow",
        "workflows_dir": str(WORKFLOWS_DIR),
    }

