import { useState } from "react";
import axios from "axios";
import { Dumbbell, AlertTriangle, Info } from "lucide-react";

function WorkoutRecommender() {
    const [formData, setFormData] = useState({
        Age: "",          // Adjusted field to match new input
        "Activity Level": "",
        Gender: ""
    });
    const [workoutPlan, setWorkoutPlan] = useState<any>(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const [activeDay, setActiveDay] = useState<string | null>(null);
    const [showDescription, setShowDescription] = useState(false);

    const activityLevels = ["Beginner", "Intermediate", "Advanced"];
    const ageGroups = ["Kids", "Teens", "Adults", "Seniors"];  // Adjusted for new age input
    const genderOptions = ["Male", "Female"];

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!formData.Age || !formData["Activity Level"] || !formData.Gender) {
            setError("Please fill all required fields");
            return;
        }

        setError("");
        setWorkoutPlan(null);
        setActiveDay(null);
        setLoading(true);

        try {
            const response = await axios.post("http://localhost:5000/api/workout", formData);
            setWorkoutPlan(response.data);
            if (response.data && Object.keys(response.data).length > 0) {
                setActiveDay(Object.keys(response.data)[0]);
            }
        } catch (err) {
            setError("Error fetching workout recommendations. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="module-container">
            <div className="module-header">
                <Dumbbell size={32} />
                <h2>Workout Recommender</h2>
            </div>

            <div className="form-description">
                <p>
                    Enter your details to get a personalized workout plan tailored to your needs.
                    <button 
                        className="info-button" 
                        onClick={() => setShowDescription(!showDescription)}
                    >
                        {showDescription ? "Hide Info" : "Show Info"}
                    </button>
                </p>

                {showDescription && (
                    <div className="additional-info">
                        <p>
                            This tool provides workout recommendations based on your age, activity level, and gender.
                            The plan is designed to help you achieve optimal fitness results safely.
                        </p>
                    </div>
                )}
            </div>

            <div className="search-container">
                <div className="form-group">
                    <label htmlFor="Age">Age</label>
                    <select
                        id="Age"
                        name="Age"
                        value={formData.Age}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select Age Group</option>
                        {ageGroups.map((age, index) => (
                            <option key={index} value={age}>{age}</option>
                        ))}
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="ActivityLevel">Activity Level</label>
                    <select
                        id="ActivityLevel"
                        name="Activity Level"
                        value={formData["Activity Level"]}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select activity level</option>
                        {activityLevels.map((level, index) => (
                            <option key={index} value={level}>{level}</option>
                        ))}
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="Gender">Gender</label>
                    <select
                        id="Gender"
                        name="Gender"
                        value={formData.Gender}
                        onChange={handleChange}
                        required
                    >
                        <option value="">Select gender</option>
                        {genderOptions.map((gender, index) => (
                            <option key={index} value={gender}>{gender}</option>
                        ))}
                    </select>
                </div>

                <button 
                    onClick={handleSubmit} 
                    disabled={loading}
                    className="search-button"
                >
                    {loading ? "Generating..." : "Get Workout Plan"}
                </button>
            </div>

            {error && <div className="error">{error}</div>}

            {loading && (
                <div className="loading-spinner">
                    <div className="spinner"></div>
                    <p>Creating your personalized workout plan...</p>
                </div>
            )}

            {workoutPlan && (
                <div className="workout-result">
                    <h3>Your Workout Plan</h3>
                    
                    <div className="day-tabs">
                        {Object.keys(workoutPlan).map((day) => (
                            <button
                                key={day}
                                className={`day-tab ${activeDay === day ? 'active' : ''}`}
                                onClick={() => setActiveDay(day)}
                            >
                                {day}
                            </button>
                        ))}
                    </div>

                    {activeDay && (
    <div className="day-workout">
        <h4>{activeDay}'s Workout</h4>
        <div className="exercise-details">
            <div className="detail-item">
                <span className="detail-label">Exercise Plan:</span>
                <span className="detail-value">{workoutPlan[activeDay]["Exercise Plan"]}</span>
            </div>
            <div className="detail-item">
                <span className="detail-label">Workout Schedule:</span>
                <span className="detail-value">{workoutPlan[activeDay]["Workout Schedule"]}</span>
            </div>
        </div>
    </div>
)}

                    <div className="workout-footer">
                        <p className="disclaimer">
                            <AlertTriangle size={16} />
                            Always consult with a fitness professional before starting a new workout regimen.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}

export default WorkoutRecommender;
