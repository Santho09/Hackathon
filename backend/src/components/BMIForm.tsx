import { useState } from "react";
import axios from "axios";
import { Scale, Activity } from "lucide-react";

function BMIForm() {
    const [formData, setFormData] = useState({ weight: "", height: "", age: "", gender: "male", activityLevel: "sedentary" });
    const [bmiResult, setBmiResult] = useState<any>(null);
    const [bmrResult, setBmrResult] = useState<any>(null);
    const [caloriesResult, setCaloriesResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError("");

        try {
            // BMI API call
            const bmiResponse = await axios.post("http://localhost:5000/api/health/bmi", {
                weight: parseFloat(formData.weight),
                height: parseFloat(formData.height),
            });
            setBmiResult(bmiResponse.data);

            // BMR API call
            const bmrResponse = await axios.post("http://localhost:5000/api/health/bmr", {
                weight: parseFloat(formData.weight),
                height: parseFloat(formData.height),
                age: parseInt(formData.age),
                gender: formData.gender,
            });
            setBmrResult(bmrResponse.data);

            // Calories API call
            const caloriesResponse = await axios.post("http://localhost:5000/api/health/calories", {
                weight: parseFloat(formData.weight),
                height: parseFloat(formData.height),
                age: parseInt(formData.age),
                gender: formData.gender,
                activityLevel: formData.activityLevel,
            });
            setCaloriesResult(caloriesResponse.data);
        } catch (error) {
            console.error("Error:", error);
            setError("An error occurred while calculating. Please check your inputs.");
        } finally {
            setLoading(false);
        }
    };

    const getBMICategory = (bmi: number) => {
        if (!bmi) return {};
        if (bmi < 18.5) return { label: "Underweight", color: "#3498db" };
        if (bmi < 25) return { label: "Normal", color: "#2ecc71" };
        if (bmi < 30) return { label: "Overweight", color: "#f39c12" };
        return { label: "Obese", color: "#e74c3c" };
    };

    const category = bmiResult ? getBMICategory(bmiResult.bmi) : null;

    return (
        <div className="module-container">
            <div className="module-header">
                <Scale size={32} />
                <h2>BMI, BMR & Calorie Calculator</h2>
            </div>
            
            <div className="form-description">
                <p>Calculate your BMI, BMR, and daily calorie intake by entering your details below.</p>
            </div>

            {error && <div className="error">{error}</div>}

            <form onSubmit={handleSubmit} className="form-grid">
                <div className="form-group">
                    <label htmlFor="weight">Weight (kg)</label>
                    <input 
                        type="number" 
                        id="weight"
                        name="weight" 
                        value={formData.weight}
                        placeholder="Enter your weight" 
                        onChange={handleChange} 
                        required 
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="height">Height (cm)</label>
                    <input 
                        type="number" 
                        id="height"
                        name="height" 
                        value={formData.height}
                        placeholder="Enter your height" 
                        onChange={handleChange} 
                        required 
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="age">Age</label>
                    <input 
                        type="number" 
                        id="age"
                        name="age" 
                        value={formData.age}
                        placeholder="Enter your age" 
                        onChange={handleChange} 
                        required 
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="gender">Gender</label>
                    <select 
                        id="gender"
                        name="gender" 
                        value={formData.gender}
                        onChange={handleChange}
                    >
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="activityLevel">Activity Level</label>
                    <select
                        id="activityLevel"
                        name="activityLevel"
                        value={formData.activityLevel}
                        onChange={handleChange}
                    >
                        <option value="sedentary">Sedentary</option>
                        <option value="light">Light</option>
                        <option value="moderate">Moderate</option>
                        <option value="active">Active</option>
                        <option value="very active">Very Active</option>
                    </select>
                </div>

                <div className="form-actions">
                    <button type="submit" disabled={loading} className="primary-button">
                        {loading ? "Calculating..." : "Calculate"}
                    </button>
                </div>
            </form>

            {(bmiResult || bmrResult || caloriesResult) && (
                <div className="results-container">
                    
{(bmiResult && typeof bmiResult.bmi === 'number') && (
    <div className="result-card">
        <h3>BMI Results</h3>
        <div className="result-value">
            <span className="value">{bmiResult.bmi.toFixed(1)}</span>
            <span className="category" style={{ backgroundColor: category?.color }}>
                {category?.label}
            </span>
        </div>
        <p className="result-description">
            Body Mass Index (BMI) is a measure of body fat based on height and weight.
        </p>
    </div>
)}


                    {bmrResult && (
                        <div className="result-card">
                            <h3>BMR Results</h3>
                            <div className="result-value">
                                <span className="value">{Math.round(bmrResult.bmr)}</span>
                                <span className="unit">calories/day</span>
                            </div>
                            <p className="result-description">
                                Basal Metabolic Rate (BMR) is the number of calories your body needs while resting.
                            </p>
                        </div>
                    )}

                    {caloriesResult && (
                        <div className="result-card">
                            <h3>Daily Calorie Requirement</h3>
                            <div className="result-value">
                                <span className="value">{caloriesResult.calories}</span>
                                <span className="unit">calories/day</span>
                            </div>
                            <p className="result-description">
                                This is the estimated number of calories you need to maintain your current weight based on your activity level.
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default BMIForm;
