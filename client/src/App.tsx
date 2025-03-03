import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import { useState } from "react";
import { Heart, Menu, X, Scale, Pill, Dumbbell, Utensils, Moon } from "lucide-react";
import BMIForm from "./components/BMIForm";
import MedicineRecommender from "./components/MedicineRecommender";
import WorkoutRecommender from "./components/WorkoutRecommender";
import DietRecommender from "./components/DietRecommender";
import SleepRecommender from "./components/SleepRecommender"; // Import the Sleep Recommender
import "./App.css";

function App() {
    const [isNavOpen, setIsNavOpen] = useState(false);

    const toggleNav = () => {
        setIsNavOpen(!isNavOpen);
    };

    return (
        <Router>
            <div className="app">
                <button className="mobile-toggle" onClick={toggleNav}>
                    {isNavOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
                
                <nav className={`sidebar ${isNavOpen ? 'open' : ''}`}>
                    <div className="logo-container">
                        <Heart size={32} className="logo-icon" />
                        <h1>HealthBuddy</h1>
                    </div>
                    
                    <ul className="nav-links">
                        <li>
                            <NavLink to="/" onClick={() => setIsNavOpen(false)} className={({isActive}) => isActive ? 'active' : ''}>
                                <Scale size={24} />
                                <span>BMI & BMR</span>
                            </NavLink>
                        </li>
                        <li>
                            <NavLink to="/medicine" onClick={() => setIsNavOpen(false)} className={({isActive}) => isActive ? 'active' : ''}>
                                <Pill size={24} />
                                <span>Medicine</span>
                            </NavLink>
                        </li>
                        <li>
                            <NavLink to="/workout" onClick={() => setIsNavOpen(false)} className={({isActive}) => isActive ? 'active' : ''}>
                                <Dumbbell size={24} />
                                <span>Workout</span>
                            </NavLink>
                        </li>
                        <li>
                            <NavLink to="/diet" onClick={() => setIsNavOpen(false)} className={({isActive}) => isActive ? 'active' : ''}>
                                <Utensils size={24} />
                                <span>Diet Plan</span>
                            </NavLink>
                        </li>
                        <li>
                            <NavLink to="/sleep" onClick={() => setIsNavOpen(false)} className={({isActive}) => isActive ? 'active' : ''}>
                                <Moon size={24} />
                                <span>Sleep</span>
                            </NavLink>
                        </li>
                    </ul>
                    
                    <div className="nav-footer">
                        <p>© 2025 HealthBuddy</p>
                    </div>
                </nav>
                
                <div className="content">
                    <div className="content-container">
                        <Routes>
                            <Route path="/" element={<BMIForm />} />
                            <Route path="/medicine" element={<MedicineRecommender />} />
                            <Route path="/workout" element={<WorkoutRecommender />} />
                            <Route path="/diet" element={<DietRecommender />} />
                            <Route path="/sleep" element={<SleepRecommender />} /> {/* Add Sleep Recommender route */}
                        </Routes>
                    </div>
                </div>
            </div>
        </Router>
    );
}

export default App;
