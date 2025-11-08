/**
 * 워크플로우를 Jupyter Notebook (.ipynb) 형식으로 변환하는 유틸리티
 */

import { Node, Connection } from "../types/editor";

interface NotebookCell {
  cell_type: "code" | "markdown";
  source: string[];
  metadata?: Record<string, any>;
  execution_count?: number | null;
  outputs?: any[];
}

interface Notebook {
  cells: NotebookCell[];
  metadata: {
    kernelspec: {
      display_name: string;
      language: string;
      name: string;
    };
    language_info: {
      name: string;
      version: string;
    };
  };
  nbformat: number;
  nbformat_minor: number;
}

/**
 * 노드 타입별 Python 코드 생성
 */
function generateNodeCode(node: Node, nodeOutputs: Record<string, any>): string {
  const nodeData = node.data as any;
  let code = "";

  switch (node.type) {
    case "hf-token": {
      const token = nodeData.token || "";
      code = `# HuggingFace 토큰 설정
import os
os.environ["HF_TOKEN"] = "${token}"

# 또는 직접 사용
from huggingface_hub import login
login(token="${token}")
print("✅ HuggingFace 토큰이 설정되었습니다.")`;
      break;
    }

    case "hf-model-downloader": {
      const modelId = nodeData.modelId || "";
      code = `# 모델 다운로드
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "${modelId}"
print(f"📥 모델 다운로드 중: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

print(f"✅ 모델 다운로드 완료: {model_id}")`;
      break;
    }

    case "hf-dataset-downloader": {
      const datasetId = nodeData.datasetId || "";
      code = `# 데이터셋 다운로드
from datasets import load_dataset

dataset_id = "${datasetId}"
print(f"📥 데이터셋 다운로드 중: {dataset_id}")

dataset = load_dataset(dataset_id)
print(f"✅ 데이터셋 다운로드 완료: {dataset_id}")
print(f"   컬럼: {list(dataset['train'].column_names) if 'train' in dataset else list(dataset[list(dataset.keys())[0]].column_names) if dataset else [])}")`;
      break;
    }

    case "dataset-preprocessor": {
      const format = nodeData.format || "instruction";
      const template = nodeData.template || "";
      const maxLength = nodeData.maxLength || 512;
      const inputColumns = nodeData.inputColumns || [];
      const outputColumns = nodeData.outputColumns || [];
      const outputSeparator = nodeData.outputSeparator || " ";

      code = `# 데이터 전처리
from transformers import AutoTokenizer

# 토크나이저 설정 (모델과 동일한 토크나이저 사용)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_dataset(examples):
    """
    데이터셋 전처리 함수
    """
    texts = []
    
    if "${format}" == "instruction":
        # Instruction 포맷: input + output
        input_cols = ${JSON.stringify(inputColumns)}
        output_cols = ${JSON.stringify(outputColumns)}
        separator = "${outputSeparator}"
        
        for i in range(len(examples[input_cols[0]])):
            input_text = separator.join([str(examples[col][i]) for col in input_cols if col in examples])
            output_text = separator.join([str(examples[col][i]) for col in output_cols if col in examples])
            
            template_str = """${template || "### Instruction:\\n{input}\\n\\n### Response:\\n{output}"}"""
            text = template_str.replace("{input}", input_text).replace("{output}", output_text)
            texts.append(text)
    
    elif "${format}" == "chat":
        # Chat 포맷
        user_col = "${nodeData.userColumn || ""}"
        assistant_col = "${nodeData.assistantColumn || ""}"
        system_col = "${nodeData.systemColumn || ""}"
        
        for i in range(len(examples[user_col])):
            messages = []
            if system_col and system_col in examples:
                messages.append({"role": "system", "content": str(examples[system_col][i])})
            messages.append({"role": "user", "content": str(examples[user_col][i])})
            messages.append({"role": "assistant", "content": str(examples[assistant_col][i])})
            
            # Chat 템플릿 적용
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
    
    else:
        # Causal LM 포맷
        text_col = "${nodeData.textColumn || ""}"
        for i in range(len(examples[text_col])):
            texts.append(str(examples[text_col][i]))
    
    # 토큰화
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=${maxLength},
        return_tensors="pt"
    )
    
    return encodings

# 데이터셋 전처리 적용
processed_dataset = dataset.map(preprocess_dataset, batched=True, remove_columns=dataset['train'].column_names)
print("✅ 데이터 전처리 완료")`;
      break;
    }

    case "dataset-splitter": {
      const trainRatio = nodeData.trainRatio || 80;
      const valRatio = nodeData.valRatio || 10;
      const testRatio = nodeData.testRatio || 10;
      const shuffle = nodeData.shuffle !== false;

      code = `# 데이터셋 분할
train_ratio = ${trainRatio / 100}
val_ratio = ${valRatio / 100}
test_ratio = ${testRatio / 100}

# 데이터셋 분할
if 'train' in processed_dataset:
    split_dataset = processed_dataset['train'].train_test_split(
        test_size=val_ratio + test_ratio,
        shuffle=${shuffle}
    )
    train_dataset = split_dataset['train']
    
    if test_ratio > 0:
        val_test_split = split_dataset['test'].train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio),
            shuffle=${shuffle}
        )
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
    else:
        val_dataset = split_dataset['test']
        test_dataset = None
else:
    # 전체 데이터셋을 분할
    split_dataset = processed_dataset.train_test_split(
        test_size=val_ratio + test_ratio,
        shuffle=${shuffle}
    )
    train_dataset = split_dataset['train']
    
    if test_ratio > 0:
        val_test_split = split_dataset['test'].train_test_split(
            test_size=test_ratio / (val_ratio + test_ratio),
            shuffle=${shuffle}
        )
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']
    else:
        val_dataset = split_dataset['test']
        test_dataset = None

print(f"✅ 데이터셋 분할 완료")
print(f"   학습: {len(train_dataset)}개")
print(f"   검증: {len(val_dataset)}개")
${testRatio > 0 ? `print(f"   테스트: {len(test_dataset)}개")` : ""}`;
      break;
    }

    case "training-config": {
      const epochs = nodeData.epochs || 3;
      const batchSize = nodeData.batchSize || 4;
      const learningRate = nodeData.learningRate || 5e-5;
      const warmupSteps = nodeData.warmupSteps || 500;
      const outputDir = nodeData.outputDir || "./output";

      code = `# 학습 설정
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="${outputDir}",
    num_train_epochs=${epochs},
    per_device_train_batch_size=${batchSize},
    per_device_eval_batch_size=${batchSize},
    learning_rate=${learningRate},
    warmup_steps=${warmupSteps},
    logging_steps=${nodeData.loggingSteps || 10},
    save_strategy="${nodeData.saveStrategy || "epoch"}",
    eval_strategy="${nodeData.evalStrategy || "epoch"}",
    load_best_model_at_end=True,
    report_to="tensorboard",
    fp16=True,  # GPU 사용 시
)

print("✅ 학습 설정 완료")`;
      break;
    }

    case "lora-config": {
      const rank = nodeData.rank || 8;
      const alpha = nodeData.alpha || 16;
      const dropout = nodeData.dropout || 0.1;
      const targetModules = nodeData.targetModules || "q_proj,v_proj";

      code = `# LoRA 설정
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=${rank},
    lora_alpha=${alpha},
    target_modules=[${targetModules.split(",").map((m: string) => `"${m.trim()}"`).join(", ")}],
    lora_dropout=${dropout},
    bias="${nodeData.bias || "none"}",
    task_type="CAUSAL_LM",
)

# LoRA 모델 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ LoRA 설정 완료")`;
      break;
    }

    case "training": {
      code = `# 모델 학습
from transformers import Trainer, DataCollatorForLanguageModeling

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# 학습 시작
print("🚀 학습 시작...")
train_result = trainer.train()

print("✅ 학습 완료!")
print(f"   최종 Loss: {train_result.training_loss:.4f}")`;
      break;
    }

    case "model-saver": {
      const savePath = nodeData.savePath || "./saved_models";
      const saveFormat = nodeData.saveFormat || "both";

      code = `# 모델 저장
save_path = "${savePath}"

if "${saveFormat}" == "both" or "${saveFormat}" == "peft":
    # LoRA 가중치만 저장
    model.save_pretrained(f"{save_path}/lora")
    tokenizer.save_pretrained(f"{save_path}/lora")
    print(f"✅ LoRA 가중치 저장 완료: {save_path}/lora")

if "${saveFormat}" == "both" or "${saveFormat}" == "merged":
    # 전체 모델 병합 및 저장
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{save_path}/merged")
    tokenizer.save_pretrained(f"{save_path}/merged")
    print(f"✅ 병합된 모델 저장 완료: {save_path}/merged")`;
      break;
    }

    default:
      code = `# ${node.type} 노드
# 이 노드는 아직 코드 생성이 지원되지 않습니다.
pass`;
  }

  return code;
}

/**
 * 워크플로우를 Jupyter Notebook으로 변환
 */
export function workflowToNotebook(nodes: Node[], connections: Connection[]): Notebook {
  // 위상 정렬로 노드 실행 순서 결정
  const executedNodes = new Set<string>();
  const nodeDependencies = new Map<string, string[]>();
  const sortedNodes: Node[] = [];

  // 각 노드의 의존성 계산
  nodes.forEach((node) => {
    const deps: string[] = [];
    connections.forEach((conn) => {
      if (conn.target === node.id) {
        deps.push(conn.source);
      }
    });
    nodeDependencies.set(node.id, deps);
  });

  // 위상 정렬 실행
  while (executedNodes.size < nodes.length) {
    let progress = false;

    for (const node of nodes) {
      if (executedNodes.has(node.id)) continue;

      const deps = nodeDependencies.get(node.id) || [];
      const allDepsExecuted = deps.every((dep) => executedNodes.has(dep));

      if (allDepsExecuted) {
        progress = true;
        sortedNodes.push(node);
        executedNodes.add(node.id);
      }
    }

    if (!progress) {
      // 순환 의존성 또는 독립 노드 처리
      const remainingNodes = nodes.filter((n) => !executedNodes.has(n.id));
      remainingNodes.forEach((node) => {
        sortedNodes.push(node);
        executedNodes.add(node.id);
      });
      break;
    }
  }

  // Notebook 셀 생성
  const cells: NotebookCell[] = [];

  // 헤더 마크다운 셀
  cells.push({
    cell_type: "markdown",
    source: [
      "# LLM Fine-tuning Pipeline\n",
      "\n",
      "이 노트북은 Mactuner 워크플로우에서 자동 생성되었습니다.\n",
      "\n",
      "## 실행 순서\n",
      "1. 필요한 라이브러리 설치\n",
      "2. 토큰 설정\n",
      "3. 모델 및 데이터셋 다운로드\n",
      "4. 데이터 전처리\n",
      "5. 모델 학습\n",
      "6. 모델 저장\n",
    ],
    metadata: {},
  });

  // 라이브러리 설치 셀
  cells.push({
    cell_type: "code",
    source: [
      "# 필요한 라이브러리 설치\n",
      "!pip install -q transformers datasets peft accelerate bitsandbytes\n",
      "!pip install -q huggingface_hub\n",
    ],
    metadata: {},
    execution_count: null,
    outputs: [],
  });

  // 각 노드별 코드 셀 생성
  const nodeOutputs: Record<string, any> = {};
  sortedNodes.forEach((node) => {
    const code = generateNodeCode(node, nodeOutputs);
    if (code.trim()) {
      cells.push({
        cell_type: "code",
        source: code.split("\n").map((line) => line + "\n"),
        metadata: {},
        execution_count: null,
        outputs: [],
      });
    }
  });

  // Notebook 구조 생성
  const notebook: Notebook = {
    cells,
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        version: "3.10.0",
      },
    },
    nbformat: 4,
    nbformat_minor: 4,
  };

  return notebook;
}

/**
 * Notebook을 파일로 다운로드
 */
export function downloadNotebook(notebook: Notebook, filename: string = "workflow.ipynb") {
  const json = JSON.stringify(notebook, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

