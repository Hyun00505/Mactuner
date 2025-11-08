/**
 * LLM 파인튜닝 워크플로우 가이드
 * 사용자가 노드 연결 방법을 쉽게 이해할 수 있도록 도와주는 컴포넌트
 */

import React from 'react';

interface WorkflowGuideProps {
  onClose?: () => void;
}

export const WorkflowGuide: React.FC<WorkflowGuideProps> = ({ onClose }) => {
  return (
    <div className="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6 max-w-2xl">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-white">🎓 LLM 파인튜닝 워크플로우 가이드</h2>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        )}
      </div>

      <div className="space-y-6 text-sm text-gray-300">
        {/* 색상 범례 */}
        <div>
          <h3 className="text-white font-semibold mb-2">🎨 연결선 색상 의미</h3>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-500"></div>
              <span>파란색: 모델 데이터 흐름</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span>초록색: 데이터셋 흐름</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
              <span>노란색: 토큰/설정 정보</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-purple-500"></div>
              <span>보라색: 학습 설정</span>
            </div>
          </div>
        </div>

        {/* 워크플로우 단계 */}
        <div>
          <h3 className="text-white font-semibold mb-3">📋 워크플로우 단계</h3>
          <div className="space-y-4">
            {/* 1단계 */}
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="text-white font-semibold mb-1">1️⃣ 모델 불러오기</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-blue-400">📂 로컬 모델</span> 또는{' '}
                <span className="text-blue-400">🤗 HF 모델 다운로드</span>
              </p>
              <p className="text-gray-500 text-xs">
                → 파란색 연결선으로 학습, 평가, 채팅 등에 연결
              </p>
            </div>

            {/* 2단계 */}
            <div className="border-l-4 border-green-500 pl-4">
              <h4 className="text-white font-semibold mb-1">2️⃣ 데이터 정의하기</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-green-400">📂 로컬 데이터셋</span> 또는{' '}
                <span className="text-green-400">🤗 HF 데이터셋 다운로드</span>
              </p>
              <p className="text-gray-500 text-xs">
                → 초록색 연결선으로 전처리, 필터, 분할에 연결
              </p>
            </div>

            {/* 3단계 */}
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="text-white font-semibold mb-1">3️⃣ 데이터 전처리 (선택)</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-purple-400">🔧 데이터 전처리</span> →{' '}
                <span className="text-purple-400">✂️ 데이터 분할</span> →{' '}
                <span className="text-purple-400">🔍 데이터 필터</span>
              </p>
              <p className="text-gray-500 text-xs">
                → 초록색 연결선으로 연결, 학습 노드로 전달
              </p>
            </div>

            {/* 4단계 */}
            <div className="border-l-4 border-yellow-500 pl-4">
              <h4 className="text-white font-semibold mb-1">4️⃣ 학습 방법 선택</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-yellow-400">🎯 LoRA 설정</span> 또는{' '}
                <span className="text-yellow-400">⚡ QLoRA 설정</span>
              </p>
              <p className="text-gray-500 text-xs">
                → 노란색/보라색 연결선으로 학습 노드에 연결
              </p>
            </div>

            {/* 5단계 */}
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="text-white font-semibold mb-1">5️⃣ 학습 파라미터 정의</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-purple-400">⚙️ 학습 설정</span> (에포크, 배치 크기 등)
              </p>
              <p className="text-gray-500 text-xs">
                → 보라색 연결선으로 학습 노드에 연결
              </p>
            </div>

            {/* 6단계 */}
            <div className="border-l-4 border-indigo-500 pl-4">
              <h4 className="text-white font-semibold mb-1">6️⃣ 학습 실행</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-indigo-400">🎓 학습 실행</span> 노드에 모든 입력 연결
              </p>
              <p className="text-gray-500 text-xs">
                입력: 모델(파란색) + 데이터셋(초록색) + 설정(보라색/노란색)
              </p>
            </div>

            {/* 7단계 */}
            <div className="border-l-4 border-teal-500 pl-4">
              <h4 className="text-white font-semibold mb-1">7️⃣ 학습 모니터링 (선택)</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-teal-400">📊 모델 평가</span> 또는{' '}
                <span className="text-teal-400">💾 체크포인트 관리</span>
              </p>
              <p className="text-gray-500 text-xs">
                → 학습된 모델을 평가하거나 최적 체크포인트 선택
              </p>
            </div>

            {/* 8단계 */}
            <div className="border-l-4 border-red-500 pl-4">
              <h4 className="text-white font-semibold mb-1">8️⃣ 모델 저장</h4>
              <p className="text-gray-400 text-xs mb-2">
                <span className="text-red-400">💿 모델 저장</span> 또는{' '}
                <span className="text-red-400">📦 GGUF 내보내기</span>
              </p>
              <p className="text-gray-500 text-xs">
                → 최종 모델을 저장하거나 양자화하여 내보내기
              </p>
            </div>
          </div>
        </div>

        {/* 팁 */}
        <div className="bg-gray-700 rounded-lg p-3">
          <h4 className="text-white font-semibold mb-2">💡 팁</h4>
          <ul className="space-y-1 text-xs text-gray-400">
            <li>• 포트에 마우스를 올리면 연결 가능 여부가 색상으로 표시됩니다</li>
            <li>• 초록색: 연결 가능 | 빨간색: 연결 불가능</li>
            <li>• 마우스 휠로 줌 인/아웃, 우측 상단 버튼으로도 제어 가능</li>
            <li>• Shift + 드래그로 캔버스 이동</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

