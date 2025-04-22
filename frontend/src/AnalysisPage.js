import React from 'react';
import { useLocation } from 'react-router-dom';
import './AnalysisPage.css';

const AnalysisPage = () => {
  const location = useLocation();
  const { topic, selectedData } = location.state || {};

  if (!topic || !selectedData) {
    return <div className="analysis-page">No data available for analysis</div>;
  }

  return (
    <div className="analysis-page">
      <h1>Analysis in Progress</h1>
      <div className="analysis-content">
        <h2>Topic: {topic.topic}</h2>
        <div className="selected-data">
          <h3>Selected Tables and Columns:</h3>
          {selectedData.map((data, index) => (
            <div key={index} className="data-item">
              <h4>Table: {data.table}</h4>
              <p>Columns: {data.columns.join(', ')}</p>
            </div>
          ))}
        </div>
        <div className="loading-message">
          <p>Processing your data and generating analysis...</p>
          <p>This may take a few moments.</p>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage; 