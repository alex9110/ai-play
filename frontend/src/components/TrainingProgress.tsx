import React, { useEffect, useState, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { LoaderSpinner } from './LoaderSpinner';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface TrainingMetrics {
  epoch: number;
  train_loss: number;
  val_loss: number;
  is_best?: boolean;
}

interface TrainingProgressProps {
  isTraining: boolean;
  onTrainingComplete?: () => void;
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  isTraining,
  onTrainingComplete
}) => {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!isTraining) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    // Connect to SSE stream
    const eventSource = new EventSource(
      `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/train/progress`
    );
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.status === 'completed') {
          eventSource.close();
          if (onTrainingComplete) {
            onTrainingComplete();
          }
          return;
        }

        if (data.epoch) {
          setMetrics((prev) => {
            const existing = prev.find((m) => m.epoch === data.epoch);
            if (existing) {
              return prev.map((m) =>
                m.epoch === data.epoch
                  ? { ...data, train_loss: data.train_loss, val_loss: data.val_loss }
                  : m
              );
            }
            return [...prev, data];
          });
          setCurrentEpoch(data.epoch);
          if (data.total_epochs) {
            setTotalEpochs(data.total_epochs);
          }
        }
      } catch (error) {
        console.error('Error parsing SSE data:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [isTraining, onTrainingComplete]);

  // Prepare chart data
  const chartData = {
    labels: metrics.map((m) => `Epoch ${m.epoch}`),
    datasets: [
      {
        label: 'Train Loss',
        data: metrics.map((m) => m.train_loss),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Validation Loss',
        data: metrics.map((m) => m.val_loss),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const
      },
      title: {
        display: true,
        text: 'Training Progress'
      }
    },
    scales: {
      y: {
        beginAtZero: false
      }
    }
  };

  if (!isTraining && metrics.length === 0) {
    return null;
  }

  const latestMetrics = metrics[metrics.length - 1];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <div className="mb-4">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Training Progress
        </h3>
        {isTraining && (
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <LoaderSpinner size="sm" />
            <span>Epoch {currentEpoch} / {totalEpochs || '?'}</span>
          </div>
        )}
        {!isTraining && latestMetrics && (
          <p className="text-sm text-green-600 font-medium">
            Training completed!
          </p>
        )}
      </div>

      {latestMetrics && (
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 p-3 rounded">
            <p className="text-xs text-gray-600">Train Loss</p>
            <p className="text-lg font-semibold text-blue-700">
              {latestMetrics.train_loss.toFixed(4)}
            </p>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <p className="text-xs text-gray-600">Val Loss</p>
            <p className="text-lg font-semibold text-green-700">
              {latestMetrics.val_loss.toFixed(4)}
            </p>
          </div>
        </div>
      )}

      {metrics.length > 0 && (
        <div className="h-64">
          <Line data={chartData} options={chartOptions} />
        </div>
      )}
    </div>
  );
};

