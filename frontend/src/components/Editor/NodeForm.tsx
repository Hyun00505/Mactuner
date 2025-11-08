/**
 * 동적 노드 파라미터 폼
 * JSON 정의에 따라 파라미터 UI를 생성합니다
 */

import React, { useState, useEffect } from 'react';
import { NodeParameter, getVisibleParameters, fetchDynamicOptions, ParameterOption } from '../../utils/nodeLoader';

interface NodeFormProps {
  parameters?: NodeParameter[];
  values: Record<string, any>;
  onChange: (parameterId: string, value: any) => void;
  onFileSelect?: (parameterId: string, file: File) => void;
}

export const NodeForm: React.FC<NodeFormProps> = ({
  parameters = [],
  values,
  onChange,
  onFileSelect,
}) => {
  const [dynamicOptions, setDynamicOptions] = useState<Record<string, ParameterOption[]>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  // 동적 옵션 로드
  useEffect(() => {
    if (!parameters || parameters.length === 0) return;
    
    const loadDynamic = async () => {
      const options: Record<string, ParameterOption[]> = {};
      
      for (const param of parameters) {
        if (param.dynamicOptions && param.apiEndpoint) {
          setLoading(prev => ({ ...prev, [param.id]: true }));
          
          try {
            const opts = await fetchDynamicOptions(param.apiEndpoint);
            options[param.id] = opts;
          } catch (error) {
            console.error(`Failed to load options for ${param.id}:`, error);
          }
          
          setLoading(prev => ({ ...prev, [param.id]: false }));
        }
      }

      if (Object.keys(options).length > 0) {
        setDynamicOptions(options);
      }
    };

    loadDynamic();
  }, [parameters]);

  // 표시할 파라미터 필터링
  const visibleParams = getVisibleParameters(parameters, values);

  return (
    <div className="space-y-3">
      {visibleParams.map(param => (
        <ParameterField
          key={param.id}
          parameter={param}
          value={values[param.id]}
          onChange={(value) => onChange(param.id, value)}
          onFileSelect={(file) => onFileSelect?.(param.id, file)}
          dynamicOptions={dynamicOptions[param.id]}
          isLoading={loading[param.id]}
        />
      ))}
    </div>
  );
};

interface ParameterFieldProps {
  parameter: NodeParameter;
  value: any;
  onChange: (value: any) => void;
  onFileSelect?: (file: File) => void;
  dynamicOptions?: ParameterOption[];
  isLoading?: boolean;
}

const ParameterField: React.FC<ParameterFieldProps> = ({
  parameter,
  value,
  onChange,
  onFileSelect,
  dynamicOptions,
  isLoading,
}) => {
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect?.(file);
      onChange(file.name);
    }
  };

  return (
    <div className="space-y-1.5">
      {/* 레이블 */}
      <label className="block text-sm font-semibold text-gray-700 leading-tight">
        {parameter.label}
        {parameter.required && <span className="text-red-500 ml-1">*</span>}
      </label>

      {/* 설명 */}
      {parameter.description && (
        <p className="text-xs text-gray-500">{parameter.description}</p>
      )}

      {/* 입력 필드 */}
      {parameter.type === 'text' && (
        <input
          type="text"
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={parameter.placeholder}
          className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />
      )}

      {parameter.type === 'password' && (
        <input
          type="password"
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={parameter.placeholder}
          className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />
      )}

      {parameter.type === 'number' && (
        <input
          type="number"
          value={value || ''}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          placeholder={parameter.placeholder}
          min={parameter.min}
          max={parameter.max}
          step={parameter.step}
          className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        />
      )}

      {parameter.type === 'textarea' && (
        <textarea
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={parameter.placeholder}
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
        />
      )}

      {parameter.type === 'select' && (
        <select
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          disabled={isLoading}
          className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 bg-white"
        >
          <option value="">{parameter.placeholder || '선택...'}</option>
          {(dynamicOptions || parameter.options || []).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      )}

      {parameter.type === 'multiselect' && (
        <div className="space-y-2">
          <select
            multiple
            value={Array.isArray(value) ? value.filter(v => v !== "" && v != null) : []}
            onChange={(e) => {
              const selected = Array.from(e.target.selectedOptions, option => option.value).filter(v => v !== "" && v != null);
              onChange(selected);
            }}
            disabled={isLoading}
            size={Math.min((dynamicOptions || parameter.options || []).length + 1, 6)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 bg-white"
          >
            {(dynamicOptions || parameter.options || []).map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {Array.isArray(value) && value.filter(v => v !== "" && v != null).length > 0 && (
            <div className="text-xs text-gray-500">
              선택됨: {value.filter(v => v !== "" && v != null).length}개
            </div>
          )}
        </div>
      )}

      {parameter.type === 'checkbox-group' && (
        <div className="space-y-2">
          <div className="border border-gray-300 rounded-lg p-2.5 max-h-48 overflow-y-auto bg-white">
            {(dynamicOptions || parameter.options || []).length === 0 ? (
              <div className="text-xs text-gray-400 text-center py-2">
                {isLoading ? "로딩 중..." : "데이터셋을 연결하고 로드해주세요"}
              </div>
            ) : (
              <div className="space-y-1.5">
                {(dynamicOptions || parameter.options || []).map((option) => {
                  const currentValue = Array.isArray(value) ? value : [];
                  const isChecked = currentValue.includes(option.value);
                  
                  return (
                    <label
                      key={option.value}
                      className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 px-1.5 py-1 rounded"
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={(e) => {
                          const currentValue = Array.isArray(value) ? value.filter(v => v !== "" && v != null) : [];
                          if (e.target.checked) {
                            onChange([...currentValue, option.value]);
                          } else {
                            onChange(currentValue.filter(v => v !== option.value));
                          }
                        }}
                        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700 leading-tight">{option.label}</span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>
          {Array.isArray(value) && value.filter(v => v !== "" && v != null).length > 0 && (
            <div className="text-xs text-gray-500">
              선택됨: {value.filter(v => v !== "" && v != null).length}개
            </div>
          )}
        </div>
      )}

      {parameter.type === 'checkbox' && (
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={value || false}
            onChange={(e) => onChange(e.target.checked)}
            className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="text-sm text-gray-600 leading-tight">{parameter.label}</span>
        </div>
      )}

      {parameter.type === 'file' && (
        <>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-semibold transition-colors"
            >
              📂 파일 선택
            </button>
            {value && (
              <div className="flex-1 px-3 py-2 bg-gray-100 border border-gray-300 rounded-lg text-sm text-gray-700 flex items-center">
                ✓ {value}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default NodeForm;

