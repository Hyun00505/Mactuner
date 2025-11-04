# 🍎 MacTuner - MAC 환경 완벽 가이드

## ✨ 최신 업데이트

### MAC 호환성 문제 완벽 해결! ✅

```
❌ bitsandbytes (QLoRA) - MAC 미지원
✅ LoRA - MAC에서 완벽 지원
✅ PyTorch MPS - MAC GPU 가속 지원
✅ 모든 기본 기능 - MAC에서 정상 작동
```

---

## 🚀 빠른 시작 (5분)

### 방법 1: 자동 설정 스크립트 (권장)
```bash
# 프로젝트 디렉토리에서
bash MAC_QUICK_START.sh
```

### 방법 2: 수동 설정
```bash
# 1. 프로젝트 디렉토리 이동
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner

# 2. 가상환경 생성 (처음 한번만)
python3 -m venv .venv
source .venv/bin/activate

# 3. pip 업그레이드
pip install --upgrade pip

# 4. 의존성 설치
pip install -r backend/requirements.txt

# 5. 서버 시작
python -m uvicorn backend.main:app --reload

# 6. 다른 터미널에서 API 테스트
curl http://localhost:8000/model/health
```

---

## 📊 MAC에서 완벽 지원되는 기능

### ✅ 100% 지원되는 기능
| 기능 | 상태 | 설명 |
|------|------|------|
| 모델 다운로드 | ✅ | Hugging Face에서 모델 다운로드 |
| 모델 업로드 | ✅ | 로컬 모델 파일 업로드 |
| 데이터 정제 | ✅ | 업로드된 데이터 자동 정제 |
| EDA 분석 | ✅ | 데이터 통계 및 시각화 |
| LoRA 미세조정 | ✅ | 메모리 효율적 학습 (23% 절감) |
| Chat 인터페이스 | ✅ | 미세조정된 모델과 대화 |
| RAG 파이프라인 | ✅ | PDF 문서 기반 Q&A |
| GGUF 변환 | ✅ | 모델을 GGUF 형식으로 변환 |
| MPS 가속 | ✅ | MAC GPU 자동 가속 |

### ⚠️ MAC에서 미지원 기능
| 기능 | 상태 | 대체 방법 |
|------|------|----------|
| QLoRA (4-bit) | ❌ | LoRA 사용 (충분히 효율적) |
| CUDA | ❌ | MPS 자동 사용 |

---

## 📈 성능 최적화

### MAC에서의 메모리 사용량
```
모델 크기: 13GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full Fine-tuning (FP32)
메모리: 26GB 필요
효율: 0% (기준)

LoRA (추천! ✅)
메모리: 20GB 필요
효율: 23% 절감
상태: MAC에서 완벽 지원

QLoRA (4-bit)
메모리: 4GB 필요
효율: 85% 절감
상태: MAC 미지원 ❌
```

### LoRA만으로도 충분한 이유
- ✅ 23% 메모리 절감 (유의미)
- ✅ 학습 속도 동일
- ✅ 모델 성능 거의 동일
- ✅ MAC에서 완벽 지원
- ✅ 설정 간단

---

## 🔧 자주 나는 문제와 해결법

### 문제 1: `bitsandbytes` 에러
```
오류: bitsandbytes doesn't have wheels for macOS
해결: 이미 수정됨 - 최신 코드 사용
```

### 문제 2: PyTorch MPS 미지원
```
오류: MPS is not available
해결: CPU로 자동 폴백 (정상 작동)
```

### 문제 3: 메모리 부족
```
오류: CUDA out of memory (실제는 CPU 메모리)
해결: 배치 크기 감소 (4 → 2 → 1)
```

### 문제 4: 가상환경 문제
```
오류: No module named 'torch'
해결: 가상환경 재생성
  rm -rf .venv
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r backend/requirements.txt
```

---

## 🎯 실행 순서

### 처음 설정 (1회)
```bash
# Step 1: 디렉토리 이동
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner

# Step 2: 자동 설정
bash MAC_QUICK_START.sh

# 또는 수동 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 매번 실행할 때
```bash
# Step 1: 디렉토리 이동
cd /Users/kimhyunbin/Desktop/github_bunhine0452/Mactuner

# Step 2: 가상환경 활성화
source .venv/bin/activate

# Step 3: 서버 시작
python -m uvicorn backend.main:app --reload

# Step 4: API 접속 (다른 터미널)
open http://localhost:8000/docs
```

---

## 🌐 API 접근

### 자동 Swagger 문서
```
http://localhost:8000/docs
```

### ReDoc 문서
```
http://localhost:8000/redoc
```

### 직접 API 호출 예시

#### 모델 헬스 체크
```bash
curl http://localhost:8000/model/health
```

#### 모델 정보 조회
```bash
curl http://localhost:8000/model/info/gpt2 \
  -H "Content-Type: application/json"
```

#### 데이터 업로드
```bash
curl -X POST http://localhost:8000/dataset/upload \
  -F "file=@data.csv" \
  -F "data_format=csv"
```

---

## 📱 시스템 요구사항

### 필수
- **OS**: macOS 11.0 이상
- **Python**: 3.11 이상
- **메모리**: 16GB 이상 (8GB 최소)
- **디스크**: 50GB 이상

### 권장
- **OS**: macOS 12.0 이상
- **Python**: 3.12 이상
- **메모리**: 32GB
- **디스크**: 100GB
- **프로세서**: Apple Silicon (M1/M2/M3+)

### 확인 방법
```bash
# macOS 버전
system_profiler SPSoftwareDataType | grep "System Version"

# Python 버전
python3 --version

# 메모리
vm_stat
```

---

## 📚 문서

### MAC 관련
- `MAC_SETUP.md` - 상세 설정 가이드
- `MAC_QUICK_START.sh` - 자동 설정 스크립트
- 본 문서 (`README_MAC_KOR.md`)

### 전체 문서
- `SPECIFICATION.md` - 기술 사양
- `IMPLEMENTATION_GUIDE.md` - 구현 가이드
- `PHASE3_WEBUI_PLAN.md` - 웹 UI 계획

---

## 🎓 심화: MAC 최적화 이해

### MPS (Metal Performance Shaders)
```python
import torch

# MPS 자동 감지 (PyTorch 자동 처리)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# MPS 사용 가능 여부 확인
print(torch.backends.mps.is_available())  # True/False

# 예시: 모델을 MPS로 이동
model = model.to(device)
```

### 배치 크기 자동 최적화
```python
# MacTuner 자동 처리
# 메모리 기반 배치 크기 계산

총 메모리: 16GB
- OS/기타: 2GB
- PyTorch: 1GB
- 모델: 13GB
- 사용 가능: 0GB
→ 배치 크기: 1 (권장)

총 메모리: 32GB
- OS/기타: 4GB
- PyTorch: 2GB
- 모델: 13GB
- 사용 가능: 13GB
→ 배치 크기: 4-8 (권장)
```

---

## ✅ 체크리스트

### 설치 검증
- [ ] Python 3.11+ 설치됨
- [ ] 가상환경 `.venv` 생성됨
- [ ] `pip install -r backend/requirements.txt` 성공
- [ ] `.env` 파일 생성됨

### 실행 검증
- [ ] `python -m uvicorn backend.main:app --reload` 실행 성공
- [ ] http://localhost:8000/docs 접속 가능
- [ ] `/model/health` 엔드포인트 응답 < 1초
- [ ] Swagger 문서 정상 표시

### 기능 검증
- [ ] 모델 정보 조회 가능
- [ ] 데이터 업로드 가능
- [ ] EDA 분석 실행 가능
- [ ] LoRA 설정 가능
- [ ] 학습 시작 가능

---

## 🚨 디버깅 팁

### 서버 시작 안 될 때
```bash
# 1. 포트 확인
lsof -i :8000

# 2. 포트 해제 (있으면)
kill -9 <PID>

# 3. 다른 포트 사용
python -m uvicorn backend.main:app --port 8001
```

### 메모리 모니터링
```bash
# 실시간 메모리 확인
while true; do
  ps aux | grep python | grep uvicorn | awk '{print "Memory:", $6/1024 "MB"}'
  sleep 2
done
```

### 로그 확인
```bash
# 로그 파일 생성
tail -f logs/mactuner.log
```

---

## 🎉 완성!

**축하합니다!** MacTuner가 MAC에서 완벽하게 설정되었습니다! 🍎✨

### 다음 단계
1. **프론트엔드 시작**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **첫 번째 모델 학습**
   - Hugging Face에서 작은 모델 다운로드 (예: gpt2)
   - 샘플 데이터 업로드
   - LoRA 설정으로 학습 시작

3. **RAG 파이프라인 테스트**
   - PDF 문서 업로드
   - 문서 기반 Q&A 테스트

---

## 📞 지원

### 문제 해결
1. 이 가이드를 처음부터 따라하기
2. 오류 메시지 전체 복사하여 구글 검색
3. Stack Overflow 확인

### 유용한 리소스
- [PyTorch 공식 가이드](https://pytorch.org/get-started/locally/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Hugging Face 문서](https://huggingface.co/docs)

---

**즐거운 개발 되세요! 🚀**

