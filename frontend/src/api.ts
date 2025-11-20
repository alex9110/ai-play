/**
 * API client for backend communication.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PredictRequest {
  n: number;
}

export interface PredictResponse {
  n: number;
  factorA: number;
  factorB: number;
  logits: number[];
  probabilities: number[];
}

export interface TrainRequest {
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  resume?: boolean;
  dataset_path?: string;
  alpha?: number;
}

export interface TrainResponse {
  message: string;
  training_started: boolean;
}

export interface GenerateDatasetRequest {
  num_samples?: number;
  min_val?: number;
  max_val?: number;
  dataset_path?: string;
}

export interface GenerateDatasetResponse {
  message: string;
  num_samples: number;
  min_val: number;
  max_val: number;
  dataset_path: string;
}

export interface ModelInfo {
  exists: boolean;
  path?: string;
  version?: string | number;
  epoch?: string | number;
  best_val_loss?: number;
  total_parameters?: number;
  trainable_parameters?: number;
  file_size_bytes?: number;
  file_size_mb?: number;
  last_trained?: string;
  model_config?: {
    vocab_size: number;
    embedding_dim: number;
    hidden_dims: number[];
    max_digits: number;
  };
  message?: string;
  error?: string;
  is_old_model?: boolean;
}

export interface TrainingStatus {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  train_accuracy?: number;
  val_accuracy?: number;
  error?: string;
}

export async function predictFactors(n: number): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ n }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }

  return response.json();
}

export async function startTraining(request: TrainRequest = {}): Promise<TrainResponse> {
  const response = await fetch(`${API_BASE_URL}/train/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Training failed');
  }

  return response.json();
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const response = await fetch(`${API_BASE_URL}/train/status`);
  
  if (!response.ok) {
    throw new Error('Failed to get training status');
  }
  
  return response.json();
}

export async function generateDataset(request: GenerateDatasetRequest = {}): Promise<GenerateDatasetResponse> {
  const response = await fetch(`${API_BASE_URL}/dataset/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Dataset generation failed');
  }

  return response.json();
}

export async function getDatasetInfo(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/dataset/info`);
  
  if (!response.ok) {
    throw new Error('Failed to get dataset info');
  }
  
  return response.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model/info`);
  
  if (!response.ok) {
    throw new Error('Failed to get model info');
  }
  
  return response.json();
}

export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}

export interface LayerStats {
  name: string;
  shape: number[];
  num_params: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  q25: number;
  q75: number;
}

export interface ModelWeightsStats {
  exists: boolean;
  model_path?: string;
  version?: string | number;
  epoch?: string | number;
  layers?: LayerStats[];
  total_parameters?: number;
  trainable_parameters?: number;
  error?: string;
  message?: string;
  is_old_model?: boolean;
}

export async function getModelWeightsStats(modelPath?: string): Promise<ModelWeightsStats> {
  const url = new URL(`${API_BASE_URL}/model/weights/stats`);
  if (modelPath) {
    url.searchParams.append('model_path', modelPath);
  }
  
  const response = await fetch(url.toString());
  
  if (!response.ok) {
    throw new Error('Failed to get model weights stats');
  }
  
  return response.json();
}

export interface DeleteModelResponse {
  success: boolean;
  message: string;
  deleted_files?: string[];
  error?: string;
}

export async function deleteModel(modelPath?: string): Promise<DeleteModelResponse> {
  const url = new URL(`${API_BASE_URL}/model/delete`);
  if (modelPath) {
    url.searchParams.append('model_path', modelPath);
  }
  
  const response = await fetch(url.toString(), {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete model');
  }
  
  return response.json();
}
