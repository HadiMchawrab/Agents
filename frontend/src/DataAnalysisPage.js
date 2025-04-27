import React from 'react';
import { useLocation } from 'react-router-dom';
import './DataAnalysisPage.css';

const DataAnalysisPage = () => {
    const location = useLocation();
    const { images_bytes } = location.state || {};

    if (!images_bytes || Object.keys(images_bytes).length === 0) {
        return (
            <div className="data-analysis-page">
                <div className="loading-message">
                    <p>No analysis graphs available...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="data-analysis-page">
            <h1>Data Analysis Results</h1>
            
            {Object.entries(images_bytes).map(([tableName, imageStrings]) => (
                <div key={tableName} className="table-section">
                    <h2>{tableName} Analysis</h2>
                    <div className="graphs-grid">
                        {imageStrings.map((base64String, index) => (
                            <div key={index} className="graph-container">
                                <img 
                                    src={`data:image/png;base64,${base64String}`}
                                    alt={`${tableName} Analysis Graph ${index + 1}`}
                                    className="analysis-graph"
                                    onError={(e) => {
                                        console.log(`Failed to load image for ${tableName}`);
                                        e.target.style.display = 'none';
                                    }}
                                />
                            </div>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
};

export default DataAnalysisPage;