import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import CSVManager from './CSVManager';
import ResultsPage from './ResultsPage';
import TopicDetailsPage from './TopicDetailsPage';
import HelpPage from './HelpPage';
import './CSVManager.css';

function App() {
  const [topics, setTopics] = useState([]);
  const [tables, setTables] = useState([]);
  const [columnsByTable, setColumnsByTable] = useState({});

  const handleProcessComplete = (result) => {
    setTopics(result.topics || []);
    setTables(result.tables || []);
    setColumnsByTable(result.columnsByTable || {});
  };

  const handleTopicSelect = (topic, tables, columnsByTable) => {
    console.log('Selected topic:', topic);
    console.log('Tables:', tables);
    console.log('Columns by Table:', columnsByTable);
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
              element={<ResultsPage topics={topics} tables={tables} columnsByTable={columnsByTable} onTopicSelect={handleTopicSelect} />} 
            />
            <Route 
              path="/topic-details" 
              element={<TopicDetailsPage />} 
            />
            <Route 
              path="/help" 
              element={<HelpPage />} 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
