import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import CSVManager from './CSVManager';
import ResultsPage from './ResultsPage';
import TopicDetailsPage from './TopicDetailsPage';
import './CSVManager.css';

function App() {
  const [topics, setTopics] = useState([]);

  const handleProcessComplete = (result) => {
    // Assuming the backend returns topics in the format:
    // [{ title: string, description: string }, ...]
    setTopics(result.topics || []);
  };

  const handleTopicSelect = (topic) => {
    // Handle topic selection - we can implement this later
    console.log('Selected topic:', topic);
  };

  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-content">
            <div className="logo">AI Consultant</div>
            <div className="nav-links">
              <Link to="/" className="nav-link">New Process</Link>
              <Link to="/help" className="nav-link">Help</Link>
            </div>
          </div>
        </nav>
        <main>
          <Routes>
            <Route path="/" element={<CSVManager onProcessComplete={handleProcessComplete} />} />
            <Route 
              path="/results" 
              element={<ResultsPage topics={topics} onTopicSelect={handleTopicSelect} />} 
            />
            <Route path="/topic-details" element={<TopicDetailsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
