"""모델 로딩 서비스"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.config import settings
from backend.utils.mac_optimization import MACOptimizer
from backend.services.device_manager import get_device_manager

logger = logging.getLogger(__name__)


class ModelService:
    """모델 로드 및 관리 서비스"""

    def __init__(self):
        # 선택된 디바이스 또는 기본값 사용
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_current_device()
        self.cache_dir = settings.MODEL_CACHE_DIR

    def load_from_hub(
        self, model_id: str, token: Optional[str] = None, progress_callback=None
    ) -> Tuple[torch.nn.Module, AutoTokenizer, Dict]:
        """Hugging Face Hub에서 모델 다운로드 및 로드"""
        try:
            # 토큰 설정
            hf_token = token or settings.HUGGINGFACE_TOKEN

            if progress_callback:
                progress_callback({"status": "loading_tokenizer", "message": "토크나이저 로드 중...", "progress": 10})

            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token if hf_token else None,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            if progress_callback:
                progress_callback({"status": "tokenizer_done", "message": "토크나이저 로드 완료", "progress": 20})

            if progress_callback:
                progress_callback({"status": "loading_model_config", "message": "모델 구성 로드 중...", "progress": 25})

            # HuggingFace 라이브러리 로그 레벨 설정 (진행 정보 출력)
            transformers_logger = logging.getLogger("transformers")
            old_level = transformers_logger.level
            transformers_logger.setLevel(logging.INFO)
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token if hf_token else None,
                device_map="auto" if self.device.type == "cpu" else None,
                torch_dtype=torch.float16 if self.device.type == "mps" else "auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            
            # 로그 레벨 복원
            transformers_logger.setLevel(old_level)

            if progress_callback:
                progress_callback({"status": "model_loaded", "message": "모델 로드 완료", "progress": 85})

            # 디바이스로 이동
            if self.device.type != "cpu" or self.device.type != "auto":
                model = model.to(self.device)
                if progress_callback:
                    progress_callback({"status": "model_moved_to_device", "message": f"모델을 {self.device}로 이동", "progress": 90})

            metadata = self._extract_metadata(model, model_id, "hub")
            
            if progress_callback:
                progress_callback({"status": "metadata_extracted", "message": "메타데이터 추출 완료", "progress": 95})
            
            return model, tokenizer, metadata

        except Exception as e:
            raise RuntimeError(f"모델 로드 실패 ({model_id}): {str(e)}")

    def load_local(self, path: str, progress_callback=None) -> Tuple[torch.nn.Module, AutoTokenizer, Dict]:
        """로컬 경로에서 모델 로드"""
        path = Path(path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"모델 경로를 찾을 수 없음: {path}")

        try:
            if progress_callback:
                progress_callback({"status": "loading_tokenizer", "message": "토크나이저 로드 중...", "progress": 10})

            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                str(path),
                trust_remote_code=True,
            )

            if progress_callback:
                progress_callback({"status": "tokenizer_done", "message": "토크나이저 로드 완료", "progress": 20})

            if progress_callback:
                progress_callback({"status": "loading_model_config", "message": "모델 구성 로드 중...", "progress": 25})

            # HuggingFace 라이브러리 로그 레벨 설정 (진행 정보 출력)
            transformers_logger = logging.getLogger("transformers")
            old_level = transformers_logger.level
            transformers_logger.setLevel(logging.INFO)
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                str(path),
                device_map="auto" if self.device.type == "cpu" else None,
                torch_dtype=torch.float16 if self.device.type == "mps" else "auto",
                trust_remote_code=True,
            )
            
            # 로그 레벨 복원
            transformers_logger.setLevel(old_level)

            if progress_callback:
                progress_callback({"status": "model_loaded", "message": "모델 로드 완료", "progress": 85})

            # 디바이스로 이동
            if self.device.type != "cpu" or self.device.type != "auto":
                model = model.to(self.device)
                if progress_callback:
                    progress_callback({"status": "model_moved_to_device", "message": f"모델을 {self.device}로 이동", "progress": 90})

            metadata = self._extract_metadata(model, str(path), "local")
            
            if progress_callback:
                progress_callback({"status": "metadata_extracted", "message": "메타데이터 추출 완료", "progress": 95})
            
            return model, tokenizer, metadata

        except Exception as e:
            raise RuntimeError(f"로컬 모델 로드 실패 ({path}): {str(e)}")

    def _extract_metadata(
        self, model: torch.nn.Module, model_id_or_path: str, source: str
    ) -> Dict:
        """모델 메타데이터 추출"""
        num_params = sum(p.numel() for p in model.parameters())

        return {
            "model_id": model_id_or_path,
            "source": source,
            "model_type": getattr(model.config, "model_type", "unknown"),
            "hidden_size": getattr(model.config, "hidden_size", 0),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", 0),
            "vocab_size": getattr(model.config, "vocab_size", 0),
            "num_parameters": num_params,
            "estimated_memory_gb": MACOptimizer.estimate_model_memory(num_params),
            "device": str(self.device),
            "dtype": str(next(model.parameters()).dtype),
        }

    def get_model_info(self, model_id: str, token: Optional[str] = None) -> Dict:
        """모델 정보 조회 (다운로드 없이)"""
        try:
            from huggingface_hub import model_info

            hf_token = token or settings.HUGGINGFACE_TOKEN
            info = model_info(model_id, token=hf_token if hf_token else None)

            return {
                "model_id": model_id,
                "downloads": info.downloads,
                "likes": info.likes,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "tags": info.tags,
            }
        except Exception as e:
            raise RuntimeError(f"모델 정보 조회 실패: {str(e)}")

    def list_local_models(self, model_dir: Optional[str] = None) -> list:
        """로컬에 있는 모델 목록 조회"""
        if model_dir is None:
            model_dir = self.cache_dir

        model_dir = Path(model_dir).expanduser()

        if not model_dir.exists():
            return []

        models = []
        
        try:
            # Hugging Face 캐시 디렉토리의 모든 항목 순회
            for item in model_dir.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # models--namespace--model-name 형식 파싱
                    model_data = self._extract_hf_model_data(item)
                    if model_data:
                        models.append(model_data)
                
                # 기타 캐시된 데이터셋 등도 포함
                elif item.is_dir() and item.name.startswith("datasets--"):
                    dataset_data = {
                        "path": str(item),
                        "model_id": item.name.replace("--", "/"),
                        "namespace": item.name.split("--")[1] if "--" in item.name else "unknown",
                        "source": "huggingface_dataset",
                        "size_gb": self._get_dir_size_gb(item),
                        "is_dataset": True,
                    }
                    models.append(dataset_data)
                
                # 직접 config.json이 있는 경우 (로컬 모델)
                elif item.is_dir() and (item / "config.json").exists() and not item.name.startswith("."):
                    model_data = {
                        "path": str(item),
                        "model_id": item.name,
                        "namespace": "local",
                        "source": "local",
                        "size_gb": self._get_dir_size_gb(item),
                        "config_present": True,
                        "model_present": (item / "pytorch_model.bin").exists() or 
                                        (item / "model.safetensors").exists() or
                                        any(item.glob("model-*.safetensors")),
                        "tokenizer_present": (item / "tokenizer.json").exists() or 
                                            (item / "tokenizer.model").exists(),
                    }
                    models.append(model_data)
        except Exception as e:
            import logging
            logging.error(f"Error scanning models: {str(e)}")

        # 프로젝트 루트의 models/ 폴더도 함께 확인
        try:
            project_root = Path(__file__).parent.parent.parent  # backend/services/../.. = project root
            models_folder = project_root / "models"
            
            if models_folder.exists():
                for item in models_folder.iterdir():
                    if item.name.startswith("."):
                        continue
                    
                    # GGUF 파일 확인 (루트 레벨)
                    if item.is_file() and item.suffix.lower() == ".gguf":
                        model_id = item.stem
                        # 중복 확인
                        if not any(m.get("model_id") == model_id for m in models):
                            models.append({
                                "path": str(item.absolute()),
                                "model_id": model_id,
                                "namespace": "local",
                                "source": "local_folder",
                                "size_gb": round(item.stat().st_size / (1024 ** 3), 2),
                                "model_type": "GGUF",
                            })
                    
                    # HuggingFace 폴더 확인 (config.json이 있는 경우)
                    elif item.is_dir() and (item / "config.json").exists():
                        model_id = item.name
                        # 중복 확인
                        if not any(m.get("model_id") == model_id for m in models):
                            models.append({
                                "path": str(item.absolute()),
                                "model_id": model_id,
                                "namespace": "local",
                                "source": "local_folder",
                                "size_gb": self._get_dir_size_gb(item),
                                "config_present": True,
                                "model_present": (item / "pytorch_model.bin").exists() or 
                                                (item / "model.safetensors").exists() or
                                                any(item.glob("model-*.safetensors")),
                                "tokenizer_present": (item / "tokenizer.json").exists() or 
                                                    (item / "tokenizer.model").exists(),
                                "model_type": "HuggingFace",
                            })
                    
                    # GGUF 폴더 확인 (폴더 안에 GGUF 파일이 있는 경우)
                    elif item.is_dir():
                        gguf_files = list(item.glob("*.gguf"))
                        if gguf_files:
                            # 가장 큰 GGUF 파일을 선택
                            largest_gguf = max(gguf_files, key=lambda f: f.stat().st_size)
                            model_id = item.name
                            # 중복 확인
                            if not any(m.get("model_id") == model_id for m in models):
                                models.append({
                                    "path": str(largest_gguf.absolute()),
                                    "model_id": model_id,
                                    "namespace": "local",
                                    "source": "local_folder",
                                    "size_gb": round(largest_gguf.stat().st_size / (1024 ** 3), 2),
                                    "model_type": "GGUF",
                                })
        except Exception as e:
            import logging
            logging.error(f"Error scanning project models folder: {str(e)}")

        return sorted(models, key=lambda x: x.get("size_gb", 0), reverse=True)

    def _extract_hf_model_data(self, model_dir: Path) -> Optional[dict]:
        """Hugging Face 캐시 디렉토리에서 모델 데이터 추출"""
        try:
            # 모델 이름 파싱: models--namespace--model-name
            dir_name = model_dir.name
            parts = dir_name.split("--")
            
            if len(parts) < 3:
                return None
            
            # namespace와 model_name 재구성
            namespace = parts[1]
            model_name = "--".join(parts[2:])
            model_id = f"{namespace}/{model_name}"
            
            # snapshots 폴더 확인
            snapshots_dir = model_dir / "snapshots"
            latest_snapshot = None
            
            # 방법 1: snapshots 폴더가 있는 경우 - config.json이 있는 가장 최신 스냅샷 찾기
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    # config.json이 있는 스냅샷 필터링
                    valid_snapshots = []
                    for snapshot in snapshots:
                        config_path = snapshot / "config.json"
                        if config_path.exists() or config_path.is_symlink():
                            valid_snapshots.append(snapshot)
                    
                    # config.json이 있는 스냅샷이 있으면 그중 가장 최신것 사용
                    if valid_snapshots:
                        latest_snapshot = max(valid_snapshots, key=lambda x: x.stat().st_mtime)
                    # config.json이 없어도 모델/토크나이저가 있으면 사용
                    elif any((s / "model.safetensors").exists() or (s / "model.safetensors").is_symlink() or
                            any(s.glob("*.gguf")) or any(s.glob("model-*.safetensors")) 
                            for s in snapshots):
                        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
            
            # 방법 2: snapshots 폴더가 없지만 refs/main이 있는 경우
            if not latest_snapshot:
                refs_main = model_dir / "refs" / "main"
                if refs_main.exists():
                    try:
                        with open(refs_main, "r") as f:
                            snapshot_hash = f.read().strip()
                        
                        potential_snapshot = model_dir / "snapshots" / snapshot_hash
                        if potential_snapshot.exists():
                            latest_snapshot = potential_snapshot
                    except:
                        pass
            
            # snapshots가 여전히 없으면 .no_exist 폴더 확인
            if not latest_snapshot:
                no_exist = model_dir / ".no_exist"
                if no_exist.exists():
                    try:
                        with open(no_exist, "r") as f:
                            content = f.read().strip()
                        if content:
                            # .no_exist가 있으면 불완전한 다운로드
                            return {
                                "path": str(model_dir),
                                "model_id": model_id,
                                "namespace": namespace,
                                "source": "huggingface_incomplete",
                                "size_gb": self._get_dir_size_gb(model_dir),
                                "config_present": False,
                                "model_present": False,
                                "tokenizer_present": False,
                                "incomplete": True,
                                "status": "불완전한 다운로드"
                            }
                    except:
                        pass
            
            if not latest_snapshot:
                # Snapshots을 찾을 수 없으면 모델 폴더 자체 확인
                if (model_dir / "config.json").exists() or (model_dir / "config.json").is_symlink():
                    latest_snapshot = model_dir
                else:
                    return None
            
            # 모델 파일 체크
            has_model = (latest_snapshot / "pytorch_model.bin").exists() or \
                       (latest_snapshot / "pytorch_model.bin").is_symlink() or \
                       (latest_snapshot / "model.safetensors").exists() or \
                       (latest_snapshot / "model.safetensors").is_symlink() or \
                       any(latest_snapshot.glob("model-*.safetensors")) or \
                       any(latest_snapshot.glob("*.gguf"))
            
            # config.json 체크
            config_path = latest_snapshot / "config.json"
            has_config = config_path.exists() or config_path.is_symlink()
            
            # 모델도 없고 config도 없으면 스킵
            if not has_model and not has_config:
                return None
            
            model_data = {
                "path": str(latest_snapshot),
                "model_id": model_id,
                "namespace": namespace,
                "source": "huggingface",
                "size_gb": self._get_dir_size_gb(model_dir),  # 전체 모델 폴더 크기
                "config_present": has_config,
                "model_present": has_model,
                "tokenizer_present": (latest_snapshot / "tokenizer.json").exists() or
                                    (latest_snapshot / "tokenizer.json").is_symlink() or
                                    (latest_snapshot / "tokenizer.model").exists() or
                                    (latest_snapshot / "tokenizer.model").is_symlink(),
            }
            
            # config.json에서 추가 정보 추출
            try:
                import json
                if has_config:
                    # Symlink 해석
                    config_file = config_path
                    if config_file.is_symlink():
                        config_file = config_file.resolve()
                    
                    if config_file.exists():
                        with open(config_file, "r") as f:
                            config = json.load(f)
                            model_data["model_type"] = config.get("model_type", "unknown")
                            model_data["hidden_size"] = config.get("hidden_size", 0)
                            model_data["num_hidden_layers"] = config.get("num_hidden_layers", 0)
                            if "vocab_size" in config:
                                model_data["vocab_size"] = config.get("vocab_size", 0)
            except Exception as e:
                # GGUF 같은 모델은 config.json이 없을 수 있음
                pass
            
            return model_data
        except Exception as e:
            import logging
            logging.error(f"Error extracting model data from {model_dir}: {str(e)}")
            return None

    def _get_dir_size_gb(self, path: Path) -> float:
        """디렉토리 크기를 GB 단위로 반환"""
        try:
            total_size = 0
            for f in path.rglob("*"):
                try:
                    if f.is_file() and not f.is_symlink():
                        total_size += f.stat().st_size
                except:
                    pass
            return round(total_size / (1024**3), 2)
        except:
            return 0.0
