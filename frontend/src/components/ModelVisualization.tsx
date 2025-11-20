import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { getModelWeightsStats, deleteModel } from '../api';
import { LoaderSpinner } from './LoaderSpinner';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface LayerStats {
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

interface ModelWeightsStats {
  exists: boolean;
  model_path?: string;
  version?: string | number;
  epoch?: string | number;
  layers?: LayerStats[];
  total_parameters?: number;
  trainable_parameters?: number;
  error?: string;
  message?: string;
}

interface ModelVisualizationProps {
  modelInfo?: any;
  onDeleteSuccess?: () => void;
}

export const ModelVisualization: React.FC<ModelVisualizationProps> = ({
  modelInfo,
  onDeleteSuccess
}) => {
  const [stats, setStats] = useState<ModelWeightsStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (modelInfo?.exists) {
      loadWeightsStats();
    }
  }, [modelInfo]);

  const loadWeightsStats = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getModelWeightsStats();
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model weights stats');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteModel = async () => {
    if (!window.confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      return;
    }

    setDeleting(true);
    setError('');
    try {
      await deleteModel();
      // Call success callback if provided
      if (onDeleteSuccess) {
        onDeleteSuccess();
      } else {
        // Fallback: reload page if no callback provided
        window.location.reload();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
      setDeleting(false);
    }
  };

  if (!modelInfo?.exists) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Model State Visualization
        </h3>
        <p className="text-sm text-gray-600">No model available for visualization</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Model State Visualization
        </h3>
        <div className="flex items-center gap-2">
          <LoaderSpinner size="sm" />
          <span className="text-sm text-gray-600">Loading model statistics...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Model State Visualization
        </h3>
        <p className="text-sm text-red-600">{error}</p>
        {stats && (stats.message || stats.error) && (
          <p className="text-xs text-gray-600 mt-2">
            {stats.message || stats.error}
          </p>
        )}
      </div>
    );
  }

  if (!stats || !stats.exists || !stats.layers) {
    const isOldModel = stats?.is_old_model;
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Model State Visualization
        </h3>
        <div className="space-y-2">
          {isOldModel ? (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
              <p className="text-sm text-yellow-800 font-medium mb-1">
                ⚠️ Incompatible Model Architecture
              </p>
              <p className="text-sm text-yellow-700">
                {stats?.message || 'This model was trained with the old regression architecture and is incompatible with the new classification system.'}
              </p>
              <p className="text-xs text-yellow-600 mt-2 mb-3">
                Solution: Please train a new model using the classification architecture.
              </p>
              <button
                onClick={handleDeleteModel}
                disabled={deleting}
                className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {deleting ? 'Deleting...' : 'Delete This Model'}
              </button>
            </div>
          ) : (
            <p className="text-sm text-gray-600">
              {stats?.message || stats?.error || 'Unable to load model statistics'}
            </p>
          )}
          {stats?.error && !isOldModel && (
            <p className="text-xs text-red-600 mt-2">
              Technical details: {stats.error}
            </p>
          )}
        </div>
      </div>
    );
  }

  // Prepare data for weight distribution chart
  const layerNames = stats.layers.map(l => {
    // Shorten layer names for display
    const parts = l.name.split('.');
    return parts.length > 1 ? parts.slice(-2).join('.') : l.name;
  });

  const meanData = stats.layers.map(l => l.mean);
  const stdData = stats.layers.map(l => l.std);

  const distributionChartData = {
    labels: layerNames,
    datasets: [
      {
        label: 'Mean Weight',
        data: meanData,
        backgroundColor: 'rgba(59, 130, 246, 0.6)',
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 1
      },
      {
        label: 'Std Dev',
        data: stdData,
        backgroundColor: 'rgba(16, 185, 129, 0.6)',
        borderColor: 'rgb(16, 185, 129)',
        borderWidth: 1
      }
    ]
  };

  const distributionChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const
      },
      title: {
        display: true,
        text: 'Weight Statistics by Layer'
      }
    },
    scales: {
      y: {
        beginAtZero: false
      }
    }
  };

  // Prepare data for weight range chart
  const rangeChartData = {
    labels: layerNames,
    datasets: [
      {
        label: 'Min',
        data: stats.layers.map(l => l.min),
        backgroundColor: 'rgba(239, 68, 68, 0.6)',
        borderColor: 'rgb(239, 68, 68)',
        borderWidth: 1
      },
      {
        label: 'Max',
        data: stats.layers.map(l => l.max),
        backgroundColor: 'rgba(34, 197, 94, 0.6)',
        borderColor: 'rgb(34, 197, 94)',
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-4">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Model State Visualization
        </h3>
        {stats.version && (
          <p className="text-sm text-gray-600">
            Model v{stats.version} • Epoch {stats.epoch} • {stats.total_parameters?.toLocaleString()} parameters
          </p>
        )}
      </div>

      {/* Layer Statistics Table */}
      <div className="mb-6 overflow-x-auto">
        <h4 className="text-lg font-semibold text-gray-700 mb-3">Layer Statistics</h4>
        <div className="max-h-64 overflow-y-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Layer</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Shape</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Params</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Mean</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Std</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Min</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Max</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {stats.layers.map((layer, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-gray-900 font-mono text-xs">{layer.name}</td>
                  <td className="px-3 py-2 text-gray-600 font-mono text-xs">
                    [{layer.shape.join(', ')}]
                  </td>
                  <td className="px-3 py-2 text-right text-gray-700">
                    {layer.num_params.toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right text-gray-700">
                    {layer.mean.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right text-gray-700">
                    {layer.std.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right text-gray-700">
                    {layer.min.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right text-gray-700">
                    {layer.max.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold text-gray-700 mb-3">Weight Distribution</h4>
          <div className="h-64">
            <Bar data={distributionChartData} options={distributionChartOptions} />
          </div>
        </div>
        <div>
          <h4 className="text-lg font-semibold text-gray-700 mb-3">Weight Range</h4>
          <div className="h-64">
            <Bar data={rangeChartData} options={distributionChartOptions} />
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <div className="bg-blue-50 p-3 rounded">
          <p className="text-xs text-gray-600">Total Parameters</p>
          <p className="text-lg font-semibold text-blue-700">
            {stats.total_parameters?.toLocaleString()}
          </p>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <p className="text-xs text-gray-600">Trainable</p>
          <p className="text-lg font-semibold text-green-700">
            {stats.trainable_parameters?.toLocaleString()}
          </p>
        </div>
        <div className="bg-purple-50 p-3 rounded">
          <p className="text-xs text-gray-600">Layers</p>
          <p className="text-lg font-semibold text-purple-700">
            {stats.layers.length}
          </p>
        </div>
        <div className="bg-orange-50 p-3 rounded">
          <p className="text-xs text-gray-600">Avg Weight</p>
          <p className="text-lg font-semibold text-orange-700">
            {stats.layers.length > 0
              ? (stats.layers.reduce((sum, l) => sum + Math.abs(l.mean), 0) / stats.layers.length).toFixed(4)
              : '0.0000'}
          </p>
        </div>
      </div>
    </div>
  );
};

