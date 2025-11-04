import { create } from "zustand";
import { NodeData, WorkflowData } from "../types";

interface WorkflowStore {
  // 상태
  currentWorkflow: WorkflowData | null;
  nodes: NodeData[];
  selectedNodeId: string | null;
  isLoading: boolean;
  error: string | null;

  // 액션
  setCurrentWorkflow: (workflow: WorkflowData) => void;
  addNode: (node: NodeData) => void;
  updateNode: (id: string, data: Partial<NodeData>) => void;
  deleteNode: (id: string) => void;
  selectNode: (id: string | null) => void;
  connectNodes: (sourceId: string, targetId: string) => void;
  clearWorkflow: () => void;
  saveWorkflow: (name: string) => void;
  loadWorkflow: (id: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useWorkflowStore = create<WorkflowStore>((set, get) => ({
  // 초기 상태
  currentWorkflow: null,
  nodes: [],
  selectedNodeId: null,
  isLoading: false,
  error: null,

  // 액션
  setCurrentWorkflow: (workflow) =>
    set({
      currentWorkflow: workflow,
      nodes: workflow.nodes,
    }),

  addNode: (node) =>
    set((state) => ({
      nodes: [...state.nodes, node],
    })),

  updateNode: (id, data) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === id ? { ...node, ...data } : node
      ),
    })),

  deleteNode: (id) =>
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
    })),

  selectNode: (id) => set({ selectedNodeId: id }),

  connectNodes: (sourceId, targetId) =>
    set((state) => ({
      nodes: state.nodes.map((node) => {
        if (node.id === sourceId) {
          return {
            ...node,
            connections: {
              ...node.connections,
              output: [...(node.connections.output || []), targetId],
            },
          };
        }
        if (node.id === targetId) {
          return {
            ...node,
            connections: {
              ...node.connections,
              input: [...(node.connections.input || []), sourceId],
            },
          };
        }
        return node;
      }),
    })),

  clearWorkflow: () =>
    set({
      currentWorkflow: null,
      nodes: [],
      selectedNodeId: null,
    }),

  saveWorkflow: (name) => {
    const state = get();
    const workflow: WorkflowData = {
      id: `workflow_${Date.now()}`,
      name,
      nodes: state.nodes,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    // 로컬 스토리지에 저장
    localStorage.setItem(
      `workflow_${workflow.id}`,
      JSON.stringify(workflow)
    );
    set({ currentWorkflow: workflow });
  },

  loadWorkflow: (id) => {
    const workflow = localStorage.getItem(`workflow_${id}`);
    if (workflow) {
      const parsed: WorkflowData = JSON.parse(workflow);
      set({
        currentWorkflow: parsed,
        nodes: parsed.nodes,
      });
    }
  },

  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}));
