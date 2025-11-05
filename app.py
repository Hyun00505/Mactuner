#!/usr/bin/env python3
"""
MacTuner 통합 서비스 실행 스크립트
백엔드와 프론트엔드를 한 번에 시작합니다.
"""

import os
import sys
import subprocess
import time
import signal
import atexit
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# 프로세스 저장소
processes = []

def cleanup():
    """모든 프로세스 종료"""
    print("\n🛑 서비스 종료 중...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print("✅ 모든 서비스 종료 완료")

def signal_handler(sig, frame):
    """신호 처리"""
    cleanup()
    sys.exit(0)

def start_backend():
    """백엔드 서비스 시작"""
    print("🔧 백엔드 시작 중... (포트 8001)")
    
    # Conda 환경 확인
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        print(f"✅ Conda 환경 감지: {conda_prefix}")
        # Conda 환경에서 python 명령 사용
        python_cmd = "python"
    else:
        # 가상환경 경로
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        
        if not venv_python.exists():
            print("❌ 가상환경을 찾을 수 없습니다.")
            print("   Conda 환경을 활성화하세요:")
            print("   conda activate MACtuner")
            print("   python app.py")
            sys.exit(1)
        python_cmd = str(venv_python)
    
    cmd = [
        python_cmd,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--reload",
        "--port", "8001",
        "--host", "0.0.0.0"
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        processes.append(proc)
        print("✅ 백엔드 시작됨")
        
        # 백엔드 출력 모니터링 (별도 스레드)
        import threading
        def read_backend_output():
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                print(f"[백엔드] {line.rstrip()}")
        
        def read_backend_error():
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                print(f"[백엔드 에러] {line.rstrip()}")
        
        threading.Thread(target=read_backend_output, daemon=True).start()
        threading.Thread(target=read_backend_error, daemon=True).start()
        
        return proc
    except Exception as e:
        print(f"❌ 백엔드 시작 실패: {e}")
        sys.exit(1)

def start_frontend():
    """프론트엔드 서비스 시작"""
    print("🎨 프론트엔드 시작 중... (포트 3000)")
    
    # npm 경로 확인
    cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--port", "3000"
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(proc)
        print("✅ 프론트엔드 시작됨")
        return proc
    except Exception as e:
        print(f"❌ 프론트엔드 시작 실패: {e}")
        print("   npm이 설치되어 있는지 확인하세요.")
        print("   cd frontend && npm install")
        sys.exit(1)

def print_banner():
    """시작 배너"""
    print("\n" + "="*60)
    print("🍎 MacTuner - 통합 서비스")
    print("="*60)
    print()

def print_info():
    """서비스 정보"""
    print("\n" + "="*60)
    print("✅ 모든 서비스가 시작되었습니다!")
    print("="*60)
    print()
    print("📱 접속 정보:")
    print("  🌐 프론트엔드:  http://localhost:3000")
    print("  🔌 백엔드 API:  http://localhost:8001")
    print("  📚 API 문서:    http://localhost:8001/docs")
    print()
    print("🎯 다음 단계:")
    print("  1. 브라우저에서 http://localhost:3000 열기")
    print("  2. Dashboard에서 기능 테스트")
    print("  3. Ctrl+C를 누르면 모든 서비스가 종료됩니다")
    print()
    print("="*60 + "\n")

def main():
    """메인 함수"""
    print_banner()
    
    # 신호 처리 (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    print("🚀 MacTuner 서비스를 시작합니다...\n")
    
    # 백엔드 시작
    print("[1/2] 백엔드 시작...")
    backend_proc = start_backend()
    time.sleep(3)
    
    # 프론트엔드 시작
    print("[2/2] 프론트엔드 시작...")
    frontend_proc = start_frontend()
    time.sleep(3)
    
    # 정보 출력
    print_info()
    
    # 프로세스 모니터링
    print("📊 서비스 모니터링 중... (Ctrl+C로 종료)")
    try:
        while True:
            # 프로세스 상태 확인
            if not backend_proc.poll() is None:
                print("⚠️  백엔드가 종료되었습니다.")
                cleanup()
                sys.exit(1)
            
            if not frontend_proc.poll() is None:
                print("⚠️  프론트엔드가 종료되었습니다.")
                cleanup()
                sys.exit(1)
            
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
