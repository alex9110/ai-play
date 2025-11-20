import React from 'react';
import { PredictResponse } from '../api';
import { LoaderSpinner } from './LoaderSpinner';

interface ResultCardProps {
  result: PredictResponse | null;
  loading?: boolean;
  error?: string;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, loading, error }) => {
  if (loading) {
    return (
      <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-md border border-gray-200">
        <LoaderSpinner size="md" text="Predicting factors..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full max-w-md p-6 bg-red-50 rounded-lg shadow-md border border-red-200">
        <h3 className="text-lg font-semibold text-red-800 mb-2">Error</h3>
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const product = result.factorA * result.factorB;
  const isCorrect = product === result.n;
  const confidence = isCorrect ? 100 : Math.max(0, 100 - Math.abs(product - result.n) / result.n * 100);

  return (
    <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-md border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Prediction Results</h3>
      
      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-600">Input Number</p>
          <p className="text-2xl font-bold text-gray-900">{result.n}</p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600">Factor A</p>
            <p className="text-xl font-semibold text-blue-600">{result.factorA}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Factor B</p>
            <p className="text-xl font-semibold text-blue-600">{result.factorB}</p>
          </div>
        </div>

        <div>
          <p className="text-sm text-gray-600">Product</p>
          <p className={`text-xl font-semibold ${isCorrect ? 'text-green-600' : 'text-orange-600'}`}>
            {product} {isCorrect ? '✓' : '≈'}
          </p>
        </div>

        <div>
          <p className="text-sm text-gray-600">Confidence</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  confidence > 80 ? 'bg-green-500' : confidence > 50 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(100, confidence)}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-700">{confidence.toFixed(1)}%</span>
          </div>
        </div>

        <div className="pt-4 border-t border-gray-200">
          <p className="text-sm text-gray-600 mb-2">Model Output</p>
          <div className="space-y-2 text-sm">
            <div>
              <span className="text-gray-500">Predicted Class Index:</span>
              <span className="ml-2 text-gray-800 font-mono">{result.factorA - 1}</span>
            </div>
            <div>
              <span className="text-gray-500">Top Class Probability:</span>
              <span className="ml-2 text-gray-800 font-mono">
                {result.probabilities && result.probabilities.length > 0
                  ? (Math.max(...result.probabilities) * 100).toFixed(2) + '%'
                  : 'N/A'}
              </span>
            </div>
            {result.probabilities && result.probabilities.length > 0 && (
              <div className="mt-2">
                <p className="text-xs text-gray-500 mb-1">Top 5 Class Probabilities:</p>
                <div className="space-y-1">
                  {result.probabilities
                    .map((prob, idx) => ({ prob, idx }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, 5)
                    .map(({ prob, idx }) => (
                      <div key={idx} className="flex items-center gap-2">
                        <span className="text-gray-600 font-mono text-xs">Class {idx}:</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-1.5 max-w-[100px]">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="text-gray-700 font-mono text-xs">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
