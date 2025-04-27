import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './AnalysisPage.css';

const AnalysisPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { topic, tables, submissionData } = location.state || {};
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    if (!submissionData) {
      setError("No submission data available");
      setIsLoading(false);
      return;
    }

    console.log('Submission data in AnalysisPage:', submissionData);
    console.log('Tables in AnalysisPage:', tables);

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();

    const pollForResults = async () => {
      try {
        const response = await fetch('http://localhost:5000/submit-data', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(submissionData),
          signal: abortControllerRef.current.signal
        });

        if (!response.ok) {
          const errorData = await response.text();
          throw new Error(`Failed to fetch analysis results: ${errorData}`);
        }

        const data = await response.json();
        console.log('Full analysis results received:', data);
        console.log('Images data:', data.images);
        console.log('Chosen models:', data.chosen_models);
        console.log('Explained models:', data.explained_models);
        console.log('Final scripts:', data.FinalScripts);
        
        // Only proceed with navigation if the component is still mounted
        if (abortControllerRef.current) {
          setIsLoading(false);
          const analysisState = {
            analysisResult: {
              ...data,
              additionalData: {
                chosen_models: data.chosen_models,
                explained_models: data.explained_models,
                final_scripts: data.FinalScripts
              }
            },
            tables: submissionData.tables,
            images_bytes: data.images
          };
          console.log('State being passed to DataAnalysisPage:', analysisState);
          navigate('/data-analysis', { state: analysisState });
        }
      } catch (err) {
        // Only update state if the error is not from aborting
        if (err.name !== 'AbortError') {
          console.error('Error in analysis:', err);
          setError(err.message);
          setIsLoading(false);
        }
      }
    };

    pollForResults();

    // Cleanup function to abort any ongoing fetch when component unmounts
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
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
          {submissionData && Object.entries(submissionData.tables).map(([tableName, columns], index) => (
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
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;