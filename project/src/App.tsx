import React, { useState, useEffect } from 'react';
import { Sun, Cloud, Thermometer, Wind, Droplets } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import * as tf from '@tensorflow/tfjs';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<number>(0);
  const [inputData, setInputData] = useState({
    temperature: 25,
    cloudCover: 20,
    windSpeed: 10,
    humidity: 60,
    time: 12
  });

  // Initialize and train a simple model on component mount
  useEffect(() => {
    async function createAndTrainModel() {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 10, inputShape: [5], activation: 'relu' }));
      model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
      
      model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
      
      // Generate some dummy training data
      const xs = tf.randomNormal([100, 5]);
      const ys = tf.randomNormal([100, 1]);
      
      await model.fit(xs, ys, { epochs: 10 });
      setModel(model);
    }
    
    createAndTrainModel();
  }, []);

  const makePrediction = () => {
    if (!model) return;
    
    const input = tf.tensor2d([[
      inputData.temperature,
      inputData.cloudCover,
      inputData.windSpeed,
      inputData.humidity,
      inputData.time
    ]]);
    
    const pred = model.predict(input) as tf.Tensor;
    setPrediction(Math.max(0, pred.dataSync()[0]));
  };

  const handleInputChange = (name: string, value: number) => {
    setInputData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const chartData = {
    labels: Array.from({ length: 24 }, (_, i) => i),
    datasets: [{
      label: 'Predicted Production (kW)',
      data: Array.from({ length: 24 }, () => Math.random() * 10),
      borderColor: 'rgb(255, 162, 0)',
      backgroundColor: 'rgba(255, 162, 0, 0.5)',
    }]
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-orange-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Solar Energy Production Forecasting
          </h1>
          <p className="text-gray-600">
            Predict solar energy production using machine learning
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6">Input Parameters</h2>
            
            <div className="space-y-6">
              <div>
                <label className="flex items-center gap-2 text-gray-700 mb-2">
                  <Thermometer className="w-5 h-5 text-orange-500" />
                  Temperature (°C)
                </label>
                <input
                  type="range"
                  min="-10"
                  max="45"
                  value={inputData.temperature}
                  onChange={(e) => handleInputChange('temperature', Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600">{inputData.temperature}°C</span>
              </div>

              <div>
                <label className="flex items-center gap-2 text-gray-700 mb-2">
                  <Cloud className="w-5 h-5 text-gray-500" />
                  Cloud Cover (%)
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={inputData.cloudCover}
                  onChange={(e) => handleInputChange('cloudCover', Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600">{inputData.cloudCover}%</span>
              </div>

              <div>
                <label className="flex items-center gap-2 text-gray-700 mb-2">
                  <Wind className="w-5 h-5 text-blue-500" />
                  Wind Speed (km/h)
                </label>
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={inputData.windSpeed}
                  onChange={(e) => handleInputChange('windSpeed', Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600">{inputData.windSpeed} km/h</span>
              </div>

              <div>
                <label className="flex items-center gap-2 text-gray-700 mb-2">
                  <Droplets className="w-5 h-5 text-blue-500" />
                  Humidity (%)
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={inputData.humidity}
                  onChange={(e) => handleInputChange('humidity', Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm text-gray-600">{inputData.humidity}%</span>
              </div>

              <button
                onClick={makePrediction}
                className="w-full bg-orange-500 text-white py-3 rounded-lg hover:bg-orange-600 transition-colors flex items-center justify-center gap-2"
              >
                <Sun className="w-5 h-5" />
                Generate Prediction
              </button>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-6">Production Forecast</h2>
            
            <div className="mb-8">
              <div className="text-center">
                <p className="text-gray-600 mb-2">Predicted Production</p>
                <div className="text-4xl font-bold text-orange-500">
                  {prediction.toFixed(2)} kW
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">24-Hour Forecast</h3>
              <Line data={chartData} options={{
                responsive: true,
                plugins: {
                  legend: {
                    position: 'top' as const,
                  },
                  title: {
                    display: false
                  }
                }
              }} />
            </div>
          </div>
        </div>

        <div className="mt-12 bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-6">About the Model</h2>
          <p className="text-gray-600">
            This solar energy production forecasting model uses TensorFlow.js to predict energy output based on environmental parameters. 
            The model takes into account temperature, cloud cover, wind speed, and humidity to generate predictions. 
            Please note that this is a demonstration model and actual production values may vary.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;