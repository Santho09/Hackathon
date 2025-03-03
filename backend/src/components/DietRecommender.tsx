import React, { useState } from 'react';
import axios from 'axios';

interface DietRecommendation {
    BMI: number;
    Daily_Caloric_Intake: number;
    Diet_Recommendation: string;
    'Weight Loss / Weight Gain Plan': string;
    'Recommended Recipes': string;
    'Total Calories': number;
}

const DietRecommender: React.FC = () => {
    const [age, setAge] = useState('');
    const [gender, setGender] = useState('');
    const [weightKg, setWeightKg] = useState('');
    const [heightCm, setHeightCm] = useState('');
    const [physicalActivity, setPhysicalActivity] = useState('');
    const [recommendation, setRecommendation] = useState<DietRecommendation | null>(null);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setError('');

            const userData = {
                Age: Number(age),
                Gender: gender,
                Weight_kg: Number(weightKg),
                Height_cm: Number(heightCm),
                Physical_Activity_Level: physicalActivity,
            };

            // Call Express API to connect with the Flask API
            const response = await axios.post('http://localhost:5000/api/diet', userData);
            setRecommendation(response.data);
        } catch (err) {
            setError('Failed to get diet recommendation. Please try again.');
        }
    };

    return (
        <div className="p-4 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
            <h2 className="text-xl font-bold text-center">Diet Recommender</h2>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label htmlFor="age" className="block text-sm font-medium text-gray-700">Age</label>
                    <input
                        type="number"
                        id="age"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        className="mt-1 block w-full p-2 border rounded-md shadow-sm"
                        required
                    />
                </div>

                <div>
                    <label htmlFor="gender" className="block text-sm font-medium text-gray-700">Gender</label>
                    <select
                        id="gender"
                        value={gender}
                        onChange={(e) => setGender(e.target.value)}
                        className="mt-1 block w-full p-2 border rounded-md shadow-sm"
                        required
                    >
                        <option value="">Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                <div>
                    <label htmlFor="weightKg" className="block text-sm font-medium text-gray-700">Weight (kg)</label>
                    <input
                        type="number"
                        id="weightKg"
                        value={weightKg}
                        onChange={(e) => setWeightKg(e.target.value)}
                        className="mt-1 block w-full p-2 border rounded-md shadow-sm"
                        required
                    />
                </div>

                <div>
                    <label htmlFor="heightCm" className="block text-sm font-medium text-gray-700">Height (cm)</label>
                    <input
                        type="number"
                        id="heightCm"
                        value={heightCm}
                        onChange={(e) => setHeightCm(e.target.value)}
                        className="mt-1 block w-full p-2 border rounded-md shadow-sm"
                        required
                    />
                </div>

                <div>
                    <label htmlFor="physicalActivity" className="block text-sm font-medium text-gray-700">Physical Activity Level</label>
                    <select
                        id="physicalActivity"
                        value={physicalActivity}
                        onChange={(e) => setPhysicalActivity(e.target.value)}
                        className="mt-1 block w-full p-2 border rounded-md shadow-sm"
                        required
                    >
                        <option value="">Select Activity Level</option>
                        <option value="Sedentary">Sedentary</option>
                        <option value="Moderate">Moderate</option>
                        <option value="Active"> Active</option>
                        
                    </select>
                </div>

                <button type="submit" className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600">
                    Get Diet Recommendation
                </button>
            </form>

            {error && <div className="text-red-500 text-center">{error}</div>}

            {recommendation && (
                <div className="mt-4 p-4 bg-green-100 rounded-lg">
                    <h3 className="text-lg font-bold">Diet Recommendation:</h3>
                    <p><strong>BMI:</strong> {recommendation.BMI ? recommendation.BMI.toFixed(2) : 'N/A'}</p>
                    <p><strong>Daily Caloric Intake:</strong> {recommendation.Daily_Caloric_Intake ? recommendation.Daily_Caloric_Intake.toFixed(2) : 'N/A'} kcal</p>
                    <p><strong>Diet Plan:</strong> {recommendation.Diet_Recommendation || 'N/A'}</p>
                    <p><strong>Weight Plan:</strong> {recommendation['Weight Loss / Weight Gain Plan'] || 'N/A'}</p>
                    <p><strong>Recommended Recipes:</strong> {recommendation['Recommended Recipes'] || 'N/A'}</p>
                    <p><strong>Total Calories (Recipe):</strong> {recommendation['Total Calories'] ? recommendation['Total Calories'].toFixed(2) : 'N/A'} kcal</p>
                </div>
            )}
        </div>
    );
};

export default DietRecommender;
