import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import CSVManager from './CSVManager';
import ResultsPage from './ResultsPage';
import TopicDetailsPage from './TopicDetailsPage';
import HelpPage from './HelpPage';
import AnalysisPage from './AnalysisPage';
import './CSVManager.css';

function App() {
  const [topics, setTopics] = useState([]);
  const [tables, setTables] = useState([]);
  const [columnsByTable, setColumnsByTable] = useState({});
  const [analyzedArticles, setAnalyzedArticles] = useState({});
  const [scrapedArticles, setScrapedArticles] = useState({});
  const [relationships, setRelationships] = useState({});
  const [explanations, setExplanations] = useState({});
  const [modelsPerTopic, setModelsPerTopic] = useState({});
  const [mlModels, setMlModels] = useState([]);

  const handleProcessComplete = (result) => {
    setTopics(result.topics || []);
    setTables(result.tables || []);
    setColumnsByTable(result.columnsByTable || {});
    setAnalyzedArticles(result.analyzedArticles || {});
    setScrapedArticles(result.scrapedArticles || {});
    setRelationships(result.relationships || {});
    setExplanations(result.explanations || {});
    setModelsPerTopic(result.modelsPerTopic || {});
    setMlModels(result.mlModels || []);
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
              element={
                <ResultsPage 
                  topics={topics}
                  tables={tables}
                  columnsByTable={columnsByTable}
                  analyzedArticles={analyzedArticles}
                  scrapedArticles={scrapedArticles}
                  relationships={relationships}
                  explanations={explanations}
                  modelsPerTopic={modelsPerTopic}
                  mlModels={mlModels}
                />
              } 
            />
            <Route 
              path="/topic-details" 
              element={<TopicDetailsPage />} 
            />
            <Route 
              path="/help" 
              element={<HelpPage />} 
            />
            <Route 
              path="/analysis" 
              element={<AnalysisPage />} 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
