import React, { useState } from 'react';

interface NumberInputProps {
  onSubmit: (value: number) => void;
  disabled?: boolean;
}

export const NumberInput: React.FC<NumberInputProps> = ({ onSubmit, disabled = false }) => {
  const [value, setValue] = useState<string>('');
  const [error, setError] = useState<string>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const num = parseInt(value, 10);
    if (isNaN(num) || num < 1) {
      setError('Please enter a valid positive integer');
      return;
    }

    onSubmit(num);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-md">
      <div className="flex flex-col gap-2">
        <div className="flex gap-2">
          <input
            type="number"
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setError('');
            }}
            placeholder="Enter a number (e.g., 2021)"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={disabled}
            min="1"
          />
          <button
            type="submit"
            disabled={disabled}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            Factorize
          </button>
        </div>
        {error && (
          <p className="text-red-500 text-sm">{error}</p>
        )}
      </div>
    </form>
  );
};

