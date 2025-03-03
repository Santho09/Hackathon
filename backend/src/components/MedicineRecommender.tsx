import { useState } from "react";
import axios from "axios";
import { Pill, AlertTriangle, Info } from "lucide-react";

function MedicineRecommender() {
    const [disease, setDisease] = useState("");
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const [showDescription, setShowDescription] = useState(false);

    const commonDiseases = [
        "Fungal skin infections", "Treatment of Dandruff", "Depression", "Nutritional deficiencies", 
        "Pain relief", " Depression Smoking addiction", "Treatment of Type 2 diabetes mellitus", "Treatment of Infectious diarrhea"
    ];

    const getRecommendation = async () => {
        if (!disease.trim()) {
            setError("Please enter a disease name");
            return;
        }

        try {
            setLoading(true);
            setError("");
            setResult(null);

            const response = await axios.post("http://localhost:5000/api/medicine", { disease });
            setResult(response.data);
        } catch (err) {
            setError("No medicine found for this disease. Please try a different condition.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="module-container">
            <div className="module-header">
                <Pill size={32} />
                <h2>Medicine Recommender</h2>
            </div>

            <div className="form-description">
                <p>
                    Enter a medical condition to get recommended medications and information.
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
                            This tool provides general medication information based on common conditions. 
                            Always consult with a healthcare professional before taking any medication.
                        </p>
                    </div>
                )}
            </div>

            <div className="search-container">
                <div className="search-input">
                    <input 
                        type="text" 
                        placeholder="Enter disease or condition" 
                        value={disease} 
                        onChange={(e) => setDisease(e.target.value)} 
                        list="diseases"
                    />
                    <datalist id="diseases">
                        {commonDiseases.map((d, index) => (
                            <option key={index} value={d} />
                        ))}
                    </datalist>
                </div>
                <button 
                    onClick={getRecommendation} 
                    disabled={loading}
                    className="search-button"
                >
                    {loading ? "Searching..." : "Get Medicine"}
                </button>
            </div>

            {error && <div className="error">{error}</div>}

            {result && (
                <div className="medicine-result">
                    <h3>Recommended Medicine</h3>
                    
                    <div className="medicine-card">
                        <div className="medicine-header">
                            <Pill size={24} />
                            <h4>{result["Medicine Name"]}</h4>
                        </div>
                        
                        <div className="medicine-details">
                            <div className="detail-item">
                                <span className="detail-label">Composition:</span>
                                <span className="detail-value">{result.Composition}</span>
                            </div>
                            
                            <div className="detail-item">
                                <span className="detail-label">Side Effects:</span>
                                <span className="detail-value">{result["Side Effects"]}</span>
                            </div>
                        </div>
                        
                        <div className="medicine-footer">
                            <p className="disclaimer">
                                <AlertTriangle size={16} />
                                Always consult a healthcare professional before taking medication.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default MedicineRecommender;