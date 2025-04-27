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

    // Function to convert bytes to data URL
    const bytesToDataUrl = (bytes) => {
        // Convert the byte array to base64
        const base64 = btoa(
            bytes.reduce((data, byte) => data + String.fromCharCode(byte), '')
        );
        return `data:image/png;base64,${base64}`;
    };

    return (
        <div className="data-analysis-page">
            <h1>Data Analysis Results</h1>
            
            {Object.entries(images_bytes).map(([tableName, imageByteArrays]) => (
                <div key={tableName} className="table-section">
                    <h2>{tableName} Analysis</h2>
                    <div className="graphs-grid">
                        {imageByteArrays.map((imageBytes, index) => (
                            <div key={index} className="graph-container">
                                <img 
                                    src={bytesToDataUrl(imageBytes)} 
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