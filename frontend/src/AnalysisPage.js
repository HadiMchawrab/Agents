import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './AnalysisPage.css';

const AnalysisPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { topic, tables, submissionData } = location.state || {};
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!submissionData) {
      setError("No submission data available");
      setIsLoading(false);
      return;
    }

    const pollForResults = async () => {
      try {
        const response = await fetch('http://localhost:5000/submit-data', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(submissionData)
        });

        if (!response.ok) {
          throw new Error('Failed to fetch analysis results');
        }

        const data = await response.json();
        setAnalysisResult(data);
        setIsLoading(false);

        // Navigate to data analysis page with the results including images_bytes
        navigate('/data-analysis', { 
          state: { 
            analysisResult: data,
            tables,
            images_bytes: data.images_bytes 
          } 
        });
      } catch (err) {
        setError(err.message);
        setIsLoading(false);
      }
    };

    pollForResults();
  }, [submissionData, navigate, tables]);

  if (!topic || !tables) {
    return <div className="analysis-page">No data available for analysis</div>;
  }

  return (
    <div className="analysis-page">
      <h1>Analysis in Progress</h1>
      <div className="analysis-content">
        <h2>Topic: {topic.topic}</h2>
        <div className="selected-data">
          <h3>Selected Tables and Columns:</h3>
          {Object.entries(tables).map(([tableName, columns], index) => (
            <div key={index} className="data-item">
              <h4>Table: {tableName}</h4>
              <p>Columns: {columns.join(', ')}</p>
            </div>
          ))}
        </div>
        <div className="analysis-status">
          {isLoading ? (
            <div className="loading-message">
              <p>Processing your data and generating analysis...</p>
              <p>This may take a few moments.</p>
              <div className="loading-spinner"></div>
            </div>
          ) : error ? (
            <div className="error-message">
              <p>Error: {error}</p>
              <button onClick={() => navigate(-1)}>Go Back</button>
            </div>
          ) : analysisResult ? (
            <div className="analysis-results">
              <h3>Analysis Results:</h3>
              <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;