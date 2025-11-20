import { useState, useEffect } from 'react';
import { NumberInput } from './components/NumberInput';
import { ResultCard } from './components/ResultCard';
import { TrainingProgress } from './components/TrainingProgress';
import { ModelVisualization } from './components/ModelVisualization';
import { LoaderSpinner } from './components/LoaderSpinner';
import {
  predictFactors,
  startTraining,
  generateDataset,
  getModelInfo,
  getDatasetInfo,
  getTrainingStatus,
  deleteModel,
  PredictResponse,
  ModelInfo
} from './api';
import './App.css';

function App() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  
  // Training form state
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [alpha, setAlpha] = useState(0.05);
  const [resume, setResume] = useState(false);
  const [numSamples, setNumSamples] = useState(10000);
  const [minVal, setMinVal] = useState(1);
  const [maxVal, setMaxVal] = useState(9999);
  const [generatingDataset, setGeneratingDataset] = useState(false);

  // Load initial data
  useEffect(() => {
    loadModelInfo();
    loadDatasetInfo();
    checkTrainingStatus();
    
    // Poll training status
    const interval = setInterval(() => {
      checkTrainingStatus();
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const loadModelInfo = async () => {
    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (err) {
      console.error('Failed to load model info:', err);
    }
  };

  const handleDeleteModel = async () => {
    if (!window.confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      return;
    }

    try {
      await deleteModel();
      // Reload model info after deletion
      await loadModelInfo();
      // Also reload dataset info in case it's related
      await loadDatasetInfo();
    } catch (err) {
      console.error('Failed to delete model:', err);
      alert(err instanceof Error ? err.message : 'Failed to delete model');
    }
  };

  const loadDatasetInfo = async () => {
    try {
      const info = await getDatasetInfo();
      setDatasetInfo(info);
    } catch (err) {
      console.error('Failed to load dataset info:', err);
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const status = await getTrainingStatus();
      // Only update if we got a valid response
      if (status) {
        setIsTraining(status.is_training);
        if (status.error) {
          setTrainingError(status.error);
          setIsTraining(false);
        }
      }
    } catch (err) {
      // If status check fails and we're in training state, keep it
      // This handles cases where training just started
      console.error('Failed to check training status:', err);
    }
  };

  const handleFactorize = async (n: number) => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await predictFactors(n);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateDataset = async () => {
    // Validation
    if (minVal > maxVal) {
      setTrainingError('Min Value must be less than or equal to Max Value');
      return;
    }
    
    setGeneratingDataset(true);
    setTrainingError('');
    
    try {
      await generateDataset({ 
        num_samples: numSamples,
        min_val: minVal,
        max_val: maxVal
      });
      await loadDatasetInfo();
      alert('Dataset generated successfully!');
    } catch (err) {
      setTrainingError(err instanceof Error ? err.message : 'Failed to generate dataset');
    } finally {
      setGeneratingDataset(false);
    }
  };

  const handleStartTraining = async () => {
    setTrainingError('');
    setIsTraining(true); // Set immediately for visual feedback
    
    try {
      await startTraining({
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        resume,
        alpha
      });
      // Training started successfully, status will be updated via polling
      await loadModelInfo();
    } catch (err) {
      setTrainingError(err instanceof Error ? err.message : 'Failed to start training');
      setIsTraining(false);
    }
  };

  const handleTrainingComplete = () => {
    setIsTraining(false);
    loadModelInfo();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 md:py-12">
        <div className="max-w-6xl mx-auto">
          <header className="text-center mb-8 md:mb-12">
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">
              Integer Factorization ML
            </h1>
            <p className="text-gray-600 text-sm md:text-base">
              Neural network-powered integer factorization prediction
            </p>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Prediction Section */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Prediction</h2>
              <div className="flex flex-col items-center gap-6">
                <NumberInput onSubmit={handleFactorize} disabled={loading} />
                <ResultCard result={result} loading={loading} error={error} />
              </div>
            </div>

            {/* Model & Dataset Info */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">System Status</h2>
              
              {modelInfo && (
                <div className="mb-4 p-4 bg-blue-50 rounded">
                  <h3 className="font-semibold text-blue-800 mb-2">Model</h3>
                  {modelInfo.exists ? (
                    <div className="text-sm text-gray-700 space-y-1">
                      {modelInfo.is_old_model && (
                        <div className="bg-yellow-50 border border-yellow-200 rounded p-2 mb-2">
                          <p className="text-xs text-yellow-800 font-medium mb-1">
                            ⚠️ Incompatible Model Architecture
                          </p>
                          <p className="text-xs text-yellow-700 mb-2">
                            This model was trained with the old regression architecture. Please train a new model.
                          </p>
                          <button
                            onClick={handleDeleteModel}
                            className="px-3 py-1 bg-red-600 text-white text-xs font-medium rounded hover:bg-red-700 transition-colors"
                          >
                            Delete This Model
                          </button>
                        </div>
                      )}
                      {modelInfo.error && !modelInfo.is_old_model && (
                        <div className="bg-red-50 border border-red-200 rounded p-2 mb-2">
                          <p className="text-xs text-red-800">{modelInfo.error}</p>
                        </div>
                      )}
                      <p>Version: {modelInfo.version}</p>
                      <p>Epoch: {modelInfo.epoch}</p>
                      <p>Best Val Loss: {modelInfo.best_val_loss?.toFixed(4)}</p>
                      {modelInfo.total_parameters && <p>Parameters: {modelInfo.total_parameters.toLocaleString()}</p>}
                      {modelInfo.file_size_mb && <p>Size: {modelInfo.file_size_mb} MB</p>}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-600">{modelInfo.message || 'No model found'}</p>
                  )}
                </div>
              )}

              {datasetInfo && (
                <div className="p-4 bg-green-50 rounded">
                  <h3 className="font-semibold text-green-800 mb-2">Dataset</h3>
                  {datasetInfo.exists ? (
                    <div className="text-sm text-gray-700 space-y-1">
                      <p>Samples: {datasetInfo.num_samples?.toLocaleString()}</p>
                      <p>Range: {datasetInfo.min_number} - {datasetInfo.max_number}</p>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-600">No dataset found</p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Training Section */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Training</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Dataset Generation */}
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-700">Generate Dataset</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Number of Samples</label>
                    <input
                      type="number"
                      value={numSamples}
                      onChange={(e) => setNumSamples(parseInt(e.target.value) || 10000)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="100"
                      max="100000"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Min Value</label>
                      <input
                        type="number"
                        value={minVal}
                        onChange={(e) => setMinVal(parseInt(e.target.value) || 1)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        min="1"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Max Value</label>
                      <input
                        type="number"
                        value={maxVal}
                        onChange={(e) => setMaxVal(parseInt(e.target.value) || 9999)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        min="1"
                      />
                    </div>
                  </div>
                  {minVal > maxVal && (
                    <div className="p-2 bg-red-50 border border-red-200 rounded text-red-600 text-sm">
                      Min Value must be ≤ Max Value
                    </div>
                  )}
                  <button
                    onClick={handleGenerateDataset}
                    disabled={generatingDataset || minVal > maxVal}
                    className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                  >
                    {generatingDataset ? <LoaderSpinner size="sm" text="Generating..." /> : 'Generate Dataset'}
                  </button>
                </div>
              </div>

              {/* Training Controls */}
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-700">Train Model</h3>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Epochs</label>
                      <input
                        type="number"
                        value={epochs}
                        onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        min="1"
                        max="1000"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-600 mb-1">Batch Size</label>
                      <input
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(parseInt(e.target.value) || 64)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        min="1"
                        max="512"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="0.0001"
                      max="1"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">Alpha (Product Loss Weight)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value) || 0.05)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-gray-500 mt-1">Weight for product consistency loss (default: 0.05)</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="resume"
                      checked={resume}
                      onChange={(e) => setResume(e.target.checked)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="resume" className="text-sm text-gray-600">Resume from checkpoint</label>
                  </div>
                  <button
                    onClick={handleStartTraining}
                    disabled={isTraining}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                  >
                    {isTraining ? (
                      <>
                        <LoaderSpinner size="sm" />
                        <span>Training in Progress...</span>
                      </>
                    ) : (
                      'Start Training'
                    )}
                  </button>
                </div>
              </div>
            </div>

            {trainingError && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-red-600 text-sm">
                {trainingError}
              </div>
            )}

            {/* Show training progress section */}
            <TrainingProgress isTraining={isTraining} onTrainingComplete={handleTrainingComplete} />
          </div>

          {/* Model Visualization Section */}
          {modelInfo?.exists && (
            <div className="mt-6">
              <ModelVisualization 
                modelInfo={modelInfo} 
                onDeleteSuccess={loadModelInfo}
              />
            </div>
          )}

          <footer className="mt-8 text-center text-sm text-gray-500">
            <p>
              Enter a number and let the ML model predict its factors.
              The model uses a neural network trained on factorization data.
            </p>
          </footer>
        </div>
      </div>
    </div>
  );
}

export default App;
