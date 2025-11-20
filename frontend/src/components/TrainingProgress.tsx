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
  const [startTime, setStartTime] = useState<number | null>(null);
  const [epochTimes, setEpochTimes] = useState<number[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!isTraining) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    // Reset state when training starts
    setStartTime(Date.now());
    setEpochTimes([]);
    setCurrentEpoch(0);
    setTotalEpochs(0);
    // Keep metrics for history, but reset current epoch tracking

    // Connect to SSE stream
    const eventSource = new EventSource(
      `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/train/progress`
    );
    eventSourceRef.current = eventSource;

    let lastEpochTime = Date.now();

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        console.log('[TrainingProgress] Received SSE data:', data);
        
        if (data.status === 'completed') {
          eventSource.close();
          if (onTrainingComplete) {
            onTrainingComplete();
          }
          return;
        }

        if (data.status === 'connected') {
          // Initial connection message
          if (data.total_epochs) {
            setTotalEpochs(data.total_epochs);
          }
          if (data.current_epoch) {
            setCurrentEpoch(data.current_epoch);
          }
          return;
        }

        if (data.status === 'error') {
          console.error('[TrainingProgress] SSE error:', data.error);
          return;
        }

        if (data.epoch) {
          const now = Date.now();
          const epochTime = now - lastEpochTime;
          lastEpochTime = now;

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
          
          // Track epoch times
          setEpochTimes((prev) => [...prev, epochTime]);
        } else if (data.status === 'training' || data.status === 'waiting') {
          // Heartbeat message
          if (data.total_epochs) {
            setTotalEpochs(data.total_epochs);
          }
          if (data.current_epoch !== undefined) {
            setCurrentEpoch(data.current_epoch);
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

  // Calculate time metrics
  const elapsedTime = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
  const avgEpochTime = epochTimes.length > 0 
    ? epochTimes.reduce((a, b) => a + b, 0) / epochTimes.length / 1000 
    : 0;
  const remainingEpochs = totalEpochs - currentEpoch;
  const estimatedTimeRemaining = avgEpochTime > 0 && remainingEpochs > 0
    ? Math.floor(avgEpochTime * remainingEpochs)
    : 0;

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    const minsRem = mins % 60;
    return `${hours}h ${minsRem}m`;
  };

  // Show component if training is active OR if we have metrics from previous training
  // Show component if training is active OR if we have metrics from previous training
  // Always show when training starts to provide immediate feedback
  if (!isTraining && metrics.length === 0) {
    return null;
  }

  const latestMetrics = metrics[metrics.length - 1];
  const progressPercent = totalEpochs > 0 && currentEpoch > 0 
    ? (currentEpoch / totalEpochs) * 100 
    : isTraining ? 0 : 100;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <div className="mb-4">
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Training Progress
        </h3>
        {isTraining && (
          <div className="space-y-2">
            <div className="flex items-center gap-4 text-sm">
              <LoaderSpinner size="sm" />
              {currentEpoch === 0 ? (
                <span className="text-blue-600 font-medium">Initializing training...</span>
              ) : (
                <span className="text-gray-600">Epoch {currentEpoch} / {totalEpochs || '?'}</span>
              )}
            </div>
            {totalEpochs > 0 ? (
              <>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${Math.max(progressPercent, 2)}%` }}
                  />
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs text-gray-600">
                  <div>
                    <span className="font-semibold">Elapsed:</span> {formatTime(elapsedTime)}
                  </div>
                  {avgEpochTime > 0 && (
                    <div>
                      <span className="font-semibold">Avg/Epoch:</span> {formatTime(Math.floor(avgEpochTime))}
                    </div>
                  )}
                  {estimatedTimeRemaining > 0 && (
                    <div>
                      <span className="font-semibold">ETA:</span> {formatTime(estimatedTimeRemaining)}
                    </div>
                  )}
                </div>
              </>
            ) : currentEpoch === 0 ? (
              <div className="text-xs text-blue-600 font-medium">
                Connecting to training server...
              </div>
            ) : (
              <div className="text-xs text-gray-500">
                Waiting for training to start...
              </div>
            )}
          </div>
        )}
        {!isTraining && latestMetrics && (
          <div className="space-y-2">
            <p className="text-sm text-green-600 font-medium">
              Training completed!
            </p>
            {elapsedTime > 0 && (
              <p className="text-xs text-gray-600">
                Total time: {formatTime(elapsedTime)}
              </p>
            )}
          </div>
        )}
      </div>

      {latestMetrics && (
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 p-3 rounded">
            <p className="text-xs text-gray-600">Train Loss</p>
            <p className="text-lg font-semibold text-blue-700">
              {latestMetrics.train_loss.toFixed(4)}
            </p>
            {latestMetrics.is_best && (
              <p className="text-xs text-blue-600 mt-1">⭐ Best</p>
            )}
          </div>
          <div className="bg-green-50 p-3 rounded">
            <p className="text-xs text-gray-600">Val Loss</p>
            <p className="text-lg font-semibold text-green-700">
              {latestMetrics.val_loss.toFixed(4)}
            </p>
            {latestMetrics.is_best && (
              <p className="text-xs text-green-600 mt-1">⭐ Best</p>
            )}
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

