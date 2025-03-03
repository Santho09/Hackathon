import React, { useState } from "react";
import axios from "axios";
import { Info, AlertTriangle } from "lucide-react";

interface SleepData {
    sleepTime: string;
    sleepInterruption: string;
}

interface Recommendation {
    CONSEQUENCES: string;
    "DEEP SLEEP": string;
    "HOURS OF SLEEP": string;
    RECOMMENDATIONS: string;
    "REM SLEEP": string;
    "SLEEP HEALTH": string;
    "SLEEP SCORE": number;
}

const SleepRecommender: React.FC = () => {
    const [sleepData, setSleepData] = useState<SleepData>({
        sleepTime: "",
        sleepInterruption: ""
    });
    const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
    const [error, setError] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const [showDescription, setShowDescription] = useState(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setSleepData((prevData) => ({
            ...prevData,
            [name]: value
        }));
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setRecommendation(null);

        try {
            const response = await axios.post("http://localhost:5000/api/sleep", {
                "SLEEP TIME": sleepData.sleepTime,
                "SLEEP INTERRUPTION": sleepData.sleepInterruption
            });
            setRecommendation(response.data);
        } catch (err) {
            setError("Failed to get sleep recommendations. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="module-container">
            <div className="module-header">
                <Info size={32} />
                <h2>Sleep Recommender</h2>
            </div>

            <div className="form-description">
                <p>
                    Enter your sleep details to get a personalized sleep recommendation.
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
                            This tool provides sleep recommendations based on your sleep time and interruptions.
                            The plan is designed to help you achieve better sleep health.
                        </p>
                    </div>
                )}
            </div>

            <form onSubmit={handleSubmit} className="search-container">
                <div className="form-group">
                    <label htmlFor="sleepTime">Sleep Time (in hours):</label>
                    <input
                        type="number"
                        id="sleepTime"
                        name="sleepTime"
                        value={sleepData.sleepTime}
                        onChange={handleChange}
                        required
                        style={{ padding: "5px", marginLeft: "10px" }}
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="sleepInterruption">Sleep Interruption (times per night):</label>
                    <input
                        type="number"
                        id="sleepInterruption"
                        name="sleepInterruption"
                        value={sleepData.sleepInterruption}
                        onChange={handleChange}
                        required
                        style={{ padding: "5px", marginLeft: "10px" }}
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="search-button"
                >
                    {loading ? "Loading..." : "Get Recommendation"}
                </button>
            </form>

            {error && <div className="error">{error}</div>}

            {loading && (
                <div className="loading-spinner">
                    <div className="spinner"></div>
                    <p>Generating your personalized sleep recommendation...</p>
                </div>
            )}

            {recommendation && (
                <div className="workout-result">
                    <h3>Your Sleep Recommendation</h3>
                    <div className="exercise-details">
                        <div className="detail-item">
                            <span className="detail-label">Sleep Score:</span>
                            <span className="detail-value">{recommendation["SLEEP SCORE"]}</span>
                        </div>
                        <div className="detail-item">
                            <span className="detail-label">Deep Sleep:</span>
                            <span className="detail-value">{recommendation["DEEP SLEEP"]}</span>
                        </div>
                        <div className="detail-item">
                            <span className="detail-label">REM Sleep:</span>
                            <span className="detail-value">{recommendation["REM SLEEP"]}</span>
                        </div>
                        <div className="detail-item">
                            <span className="detail-label">Hours of Sleep:</span>
                            <span className="detail-value">{recommendation["HOURS OF SLEEP"]}</span>
                        </div>
                        <div className="detail-item">
                            <span className="detail-label">Sleep Health:</span>
                            <span className="detail-value">{recommendation["SLEEP HEALTH"]}</span>
                        </div>
                        <div className="detail-item">
                            <span className="detail-label">Recommendations:</span>
                            <span className="detail-value">{recommendation.RECOMMENDATIONS}</span>
                        </div>
                    </div>

                    <div className="workout-footer">
                        <p className="disclaimer">
                            <AlertTriangle size={16} />
                            Always consult a sleep specialist for accurate diagnosis and recommendations.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SleepRecommender;
