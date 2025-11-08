/**
 * 학습 진행 상황 패널
 * 학습이 시작되면 콘솔 출력 위에 표시되는 학습 진행 상황 모니터링 컴포넌트
 */

import React, { useEffect, useRef } from "react";
import { ExecutionLog, Node } from "../../types/editor";

interface TrainingProgressPanelProps {
  executionLog: ExecutionLog | null;
  nodes: Node[];
}

export const TrainingProgressPanel: React.FC<TrainingProgressPanelProps> = ({ executionLog, nodes }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);

  // 학습 노드 찾기
  const trainingNodeExec = executionLog?.nodeExecutions.find(
    (exec) => exec.nodeId && executionLog.nodeExecutions.some((e) => e.nodeId === exec.nodeId && e.trainingLogs && e.trainingLogs.length > 0)
  );

  // 학습 로그가 있는 노드 찾기
  const trainingExecutions = executionLog?.nodeExecutions.filter((exec) => exec.trainingLogs && exec.trainingLogs.length > 0) || [];

  // 학습 노드가 실행 중인지 확인 (노드 타입으로 확인)
  const trainingNodeRunning = executionLog?.nodeExecutions.some((exec) => {
    const node = nodes.find((n) => n.id === exec.nodeId);
    return exec.status === "running" && node?.type === "training";
  }) || false;

  // 학습이 진행 중인지 확인
  const isTraining = trainingExecutions.some((exec) => exec.status === "running" || exec.status === "completed") || trainingNodeRunning;

  // Loss 데이터 추출 (training 상태인 로그만)
  const lossData = trainingExecutions.flatMap((exec) =>
    (exec.trainingLogs || [])
      .filter((log) => log.data?.loss !== undefined && log.data?.status === "training")
      .map((log) => ({
        step: log.data?.step || 0,
        loss: log.data?.loss || 0,
        epoch: log.data?.epoch || 0,
        timestamp: log.timestamp,
      }))
      .sort((a, b) => a.step - b.step)
  );

  // Eval Loss 데이터 추출
  const evalLossData = trainingExecutions.flatMap((exec) =>
    (exec.trainingLogs || [])
      .filter((log) => log.data?.eval_loss !== undefined)
      .map((log) => ({
        step: log.data?.step || 0,
        eval_loss: log.data?.eval_loss || 0,
        epoch: log.data?.epoch || 0,
        timestamp: log.timestamp,
      }))
      .sort((a, b) => a.step - b.step)
  );

  // 최신 정보 (training 상태인 로그 중에서)
  const latestLog = executionLog?.nodeExecutions
    .flatMap((exec) => exec.trainingLogs || [])
    .filter((log) => log.data?.status === "training" || log.data?.status === "completed")
    .sort((a, b) => b.timestamp - a.timestamp)[0];

  // Loss 곡선 그리기
  useEffect(() => {
    if (!chartRef.current || lossData.length === 0) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // 캔버스 크기 설정
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;
    const padding = 40;

    // 캔버스 클리어
    ctx.clearRect(0, 0, width, height);

    if (lossData.length === 0) return;

    // Loss 값 범위 계산
    const losses = lossData.map((d) => d.loss);
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const lossRange = maxLoss - minLoss || 1;

    // Eval Loss 값 범위 계산
    const evalLosses = evalLossData.map((d) => d.eval_loss);
    const minEvalLoss = evalLosses.length > 0 ? Math.min(...evalLosses) : minLoss;
    const maxEvalLoss = evalLosses.length > 0 ? Math.max(...evalLosses) : maxLoss;
    const evalLossRange = maxEvalLoss - minEvalLoss || 1;

    // 전체 범위
    const overallMin = Math.min(minLoss, minEvalLoss);
    const overallMax = Math.max(maxLoss, maxEvalLoss);
    const overallRange = overallMax - overallMin || 1;

    // 그리드 그리기
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (height - padding * 2) * (i / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // X축 (Step)
    const maxStep = Math.max(...lossData.map((d) => d.step), 1);
    const stepWidth = (width - padding * 2) / maxStep;

    // Loss 곡선 그리기
    if (lossData.length > 0) {
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      lossData.forEach((point, idx) => {
        const x = padding + (point.step / maxStep) * (width - padding * 2);
        const y = height - padding - ((point.loss - overallMin) / overallRange) * (height - padding * 2);
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    // Eval Loss 곡선 그리기
    if (evalLossData.length > 0) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.beginPath();
      evalLossData.forEach((point, idx) => {
        const x = padding + (point.step / maxStep) * (width - padding * 2);
        const y = height - padding - ((point.eval_loss - overallMin) / overallRange) * (height - padding * 2);
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    // Y축 레이블
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= 5; i++) {
      const value = overallMax - (overallRange * i) / 5;
      const y = padding + (height - padding * 2) * (i / 5);
      ctx.fillText(value.toFixed(3), padding - 5, y + 3);
    }

    // X축 레이블
    ctx.textAlign = "center";
    ctx.fillText("0", padding, height - padding + 15);
    if (maxStep > 0) {
      ctx.fillText(maxStep.toString(), width - padding, height - padding + 15);
    }
  }, [lossData, evalLossData]);

  // 학습이 진행 중이 아니면 표시하지 않음
  // 학습 로그가 있거나 학습이 진행 중이거나 완료된 경우에만 표시
  // 학습 노드가 실행 중이면 로그가 없어도 표시
  // 진행률 정보가 있으면 표시
  const hasProgressInfo = latestLog?.data?.total_steps !== null && latestLog?.data?.total_steps !== undefined && latestLog?.data?.total_steps > 0;
  const shouldShow = isTraining || trainingNodeRunning || hasProgressInfo || lossData.length > 0 || evalLossData.length > 0;
  
  if (!shouldShow) {
    return null;
  }

  // 학습이 시작되었지만 아직 로그가 없는 경우 (초기 상태)
  const isInitializing = trainingNodeRunning && lossData.length === 0;

  // 시간 포맷팅 함수
  const formatTime = (seconds: number | null | undefined): string => {
    if (seconds === null || seconds === undefined || seconds < 0) return "N/A";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    } else {
      return `${minutes}:${secs.toString().padStart(2, "0")}`;
    }
  };

  // 진행률 바 생성 함수
  const renderProgressBar = (percent: number, current: number, total: number | null | undefined): string => {
    if (total === null || total === undefined || total === 0) return "";
    const barLength = 20;
    const filled = Math.floor((percent / 100) * barLength);
    const empty = barLength - filled;
    return `${percent.toFixed(0)}%|${"█".repeat(filled)}${" ".repeat(empty)}| ${current}/${total}`;
  };

  // 최신 로그에서 진행률 정보 추출
  const progressPercent = latestLog?.data?.progress_percent || 0;
  const currentStep = latestLog?.data?.step || 0;
  const totalSteps = latestLog?.data?.total_steps || null;
  const elapsedTime = latestLog?.data?.elapsed_time || null;
  const etaSeconds = latestLog?.data?.eta_seconds || null;
  const timePerStep = latestLog?.data?.time_per_step || null;

  return (
    <div className="mb-4 bg-gradient-to-br from-gray-900 to-gray-800 rounded-lg shadow-xl border border-gray-700 overflow-hidden">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-blue-900 to-purple-900 px-4 py-3 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎓</span>
            <div>
              <h3 className="text-sm font-bold text-white">학습 진행 상황</h3>
              <p className="text-xs text-gray-300">Training Progress Monitor</p>
            </div>
          </div>
          {latestLog && (
            <div className="flex items-center gap-4 text-xs">
              {latestLog.data?.status === "training" && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-green-400 font-semibold">학습 중</span>
                </div>
              )}
              {latestLog.data?.status === "completed" && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span className="text-blue-400 font-semibold">완료</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="p-4 space-y-4">
        {/* 진행률 바 (tqdm 스타일) */}
        {latestLog && totalSteps && totalSteps > 0 && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-semibold text-gray-300">학습 진행률</h4>
              <span className="text-xs text-gray-400 font-mono">
                {progressPercent.toFixed(1)}%
              </span>
            </div>
            <div className="space-y-2">
              {/* 진행률 바 */}
              <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                  style={{ width: `${Math.min(progressPercent, 100)}%` }}
                />
              </div>
              {/* tqdm 스타일 정보 */}
              <div className="text-xs font-mono text-gray-300 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-gray-400">진행률:</span>
                  <span className="text-blue-400">{renderProgressBar(progressPercent, currentStep, totalSteps)}</span>
                </div>
                {elapsedTime !== null && (
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400">경과 시간:</span>
                    <span className="text-green-400">[{formatTime(elapsedTime)}</span>
                    {etaSeconds !== null && (
                      <>
                        <span className="text-gray-400">&lt;</span>
                        <span className="text-yellow-400">{formatTime(etaSeconds)}</span>
                      </>
                    )}
                    <span className="text-green-400">]</span>
                  </div>
                )}
                {timePerStep !== null && timePerStep > 0 && (
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400">Step당 시간:</span>
                    <span className="text-cyan-400">{timePerStep.toFixed(2)}s/it</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 현재 상태 정보 */}
        {(latestLog || isInitializing) && (
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Step</div>
              <div className="text-lg font-bold text-blue-400">{latestLog?.data?.step || 0}</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Epoch</div>
              <div className="text-lg font-bold text-purple-400">
                {latestLog?.data?.epoch !== undefined ? latestLog.data.epoch.toFixed(2) : "0.00"}
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Train Loss</div>
              <div className="text-lg font-bold text-yellow-400">
                {latestLog?.data?.loss !== undefined ? latestLog.data.loss.toFixed(4) : isInitializing ? "준비 중..." : "N/A"}
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Eval Loss</div>
              <div className="text-lg font-bold text-green-400">
                {latestLog?.data?.eval_loss !== undefined ? latestLog.data.eval_loss.toFixed(4) : isInitializing ? "대기 중..." : "N/A"}
              </div>
            </div>
          </div>
        )}

        {/* Loss 곡선 그래프 */}
        {lossData.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-gray-300">Loss 곡선</h4>
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-blue-500"></div>
                  <span className="text-gray-400">Train Loss</span>
                </div>
                {evalLossData.length > 0 && (
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-0.5 bg-green-500"></div>
                    <span className="text-gray-400">Eval Loss</span>
                  </div>
                )}
              </div>
            </div>
            <canvas ref={chartRef} className="w-full h-48 bg-gray-900 rounded" style={{ width: "100%", height: "192px" }}></canvas>
          </div>
        )}

        {/* 최근 로그 */}
        {trainingExecutions.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
            <h4 className="text-xs font-semibold text-gray-300 mb-2">최근 업데이트</h4>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {trainingExecutions
                .flatMap((exec) => exec.trainingLogs || [])
                .sort((a, b) => b.timestamp - a.timestamp)
                .slice(0, 5)
                .map((log, idx) => (
                  <div key={idx} className="text-xs text-gray-400 font-mono">
                    <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString("ko-KR")}]</span>{" "}
                    <span className="text-gray-300">{log.message}</span>
                    {log.data?.loss !== undefined && (
                      <span className="text-yellow-400 ml-2">Loss: {log.data.loss.toFixed(4)}</span>
                    )}
                    {log.data?.eval_loss !== undefined && (
                      <span className="text-green-400 ml-2">Eval Loss: {log.data.eval_loss.toFixed(4)}</span>
                    )}
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

