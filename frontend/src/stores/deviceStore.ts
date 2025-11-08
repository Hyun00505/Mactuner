/**
 * 디바이스 관리 Store (Zustand)
 * 선택된 GPU/CPU 상태를 전역으로 관리합니다.
 */

import { create } from 'zustand';

export type DeviceType = 'mps' | 'cuda' | 'cpu';

export interface Device {
  type: DeviceType;
  name: string;
  is_available: boolean;
  memory_total?: number;
  memory_allocated?: number;
  memory_reserved?: number;
  compute_capability?: string;
}

export interface MemoryInfo {
  device: string;
  allocated?: number;
  reserved?: number;
  total?: number;
  available?: number;
}

interface DeviceStore {
  // 상태
  availableDevices: Device[];
  selectedDevice: DeviceType | null;
  currentDevice: string | null;
  memoryInfo: MemoryInfo | null;
  isLoading: boolean;
  error: string | null;
  
  // 액션
  fetchAvailableDevices: () => Promise<void>;
  selectDevice: (deviceType: DeviceType) => Promise<boolean>;
  autoSelectDevice: () => Promise<void>;
  fetchMemoryInfo: () => Promise<void>;
  clearCache: () => Promise<void>;
  resetError: () => void;
}

export const useDeviceStore = create<DeviceStore>((set, get) => ({
  // 초기 상태
  availableDevices: [],
  selectedDevice: null,
  currentDevice: null,
  memoryInfo: null,
  isLoading: false,
  error: null,
  
  // 사용 가능한 모든 디바이스 조회
  fetchAvailableDevices: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch('http://localhost:8001/device/devices/available');
      if (!response.ok) throw new Error('디바이스 조회 실패');
      
      const data = await response.json();
      set({ 
        availableDevices: data.devices,
        isLoading: false,
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '디바이스 조회 중 오류 발생';
      set({ 
        error: errorMessage,
        isLoading: false,
      });
    }
  },
  
  // 특정 디바이스 선택
  selectDevice: async (deviceType: DeviceType) => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch(
        `http://localhost:8001/device/devices/select/${deviceType}`,
        { method: 'POST' }
      );
      
      if (!response.ok) throw new Error('디바이스 선택 실패');
      
      const data = await response.json();
      set({ 
        selectedDevice: deviceType,
        currentDevice: data.current_device,
        isLoading: false,
      });
      
      // 메모리 정보 업데이트
      get().fetchMemoryInfo();
      
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '디바이스 선택 중 오류 발생';
      set({ 
        error: errorMessage,
        isLoading: false,
      });
      return false;
    }
  },
  
  // 최적 디바이스 자동 선택
  autoSelectDevice: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch(
        'http://localhost:8001/device/devices/auto-select',
        { method: 'POST' }
      );
      
      if (!response.ok) throw new Error('자동 선택 실패');
      
      const data = await response.json();
      set({ 
        selectedDevice: data.selected_device as DeviceType,
        currentDevice: data.current_device,
        isLoading: false,
      });
      
      // 메모리 정보 업데이트
      get().fetchMemoryInfo();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '자동 선택 중 오류 발생';
      set({ 
        error: errorMessage,
        isLoading: false,
      });
    }
  },
  
  // 메모리 정보 조회
  fetchMemoryInfo: async () => {
    try {
      const response = await fetch('http://localhost:8001/device/devices/memory');
      if (!response.ok) throw new Error('메모리 정보 조회 실패');
      
      const data = await response.json();
      set({ memoryInfo: data });
    } catch (error) {
      // 메모리 정보 조회 실패는 무시 (캐시 정리할 때 재시도)
      console.warn('메모리 정보 조회 실패:', error);
    }
  },
  
  // 캐시 메모리 정리
  clearCache: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch(
        'http://localhost:8001/device/devices/clear-cache',
        { method: 'POST' }
      );
      
      if (!response.ok) throw new Error('캐시 정리 실패');
      
      set({ isLoading: false });
      
      // 메모리 정보 업데이트
      await get().fetchMemoryInfo();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '캐시 정리 중 오류 발생';
      set({ 
        error: errorMessage,
        isLoading: false,
      });
    }
  },
  
  // 에러 메시지 초기화
  resetError: () => set({ error: null }),
}));


