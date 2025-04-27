import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './DataAnalysisPage.css';

const DataAnalysisPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { analysisResult, tables, images_bytes } = location.state || {};

    const handleRetry = () => {
        navigate(-1);
    };

    // Check for overall analysis result first
    if (!analysisResult) {
        return (
            <div className="data-analysis-page">
                <div className="error-container">
                    <h2>Error</h2>
                    <p>No analysis results available</p>
                    <button onClick={handleRetry} className="retry-button">
                        Retry Analysis
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="data-analysis-page">
            <h1>Data Analysis Results</h1>

            {/* Analysis Status */}
            <div className="analysis-status">
                <h2>Analysis Status</h2>
                <p>{analysisResult.message || "Analysis completed"}</p>
                {analysisResult.status && (
                    <p className={`status-badge ${analysisResult.status}`}>
                        Status: {analysisResult.status}
                    </p>
                )}
            </div>

            {/* Selected Data Summary */}
            {tables && Object.keys(tables).length > 0 && (
                <div className="selected-data">
                    <h2>Analyzed Data</h2>
                    {Object.entries(tables).map(([tableName, columns], index) => (
                        <div key={index} className="data-item">
                            <h3>{tableName}</h3>
                            <p>Analyzed columns: {columns.join(', ')}</p>
                        </div>
                    ))}
                </div>
            )}

            {/* Additional Analysis Data */}
            {analysisResult.additionalData && (
                <div className="additional-data">
                    <h2>Additional Analysis Information</h2>
                    <pre>{JSON.stringify(analysisResult.additionalData, null, 2)}</pre>
                </div>
            )}

            {/* Visualization Results */}
            {images_bytes && Object.entries(images_bytes).map(([tableName, imageStrings]) => (
                <div key={tableName} className="table-section">
                    <h2>{tableName} Visualizations</h2>
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

            {/* Show retry button if no visualizations */}
            {(!images_bytes || Object.keys(images_bytes).length === 0) && (
                <div className="no-visualizations">
                    <p>No visualization graphs were generated during the analysis.</p>
                    <button onClick={handleRetry} className="retry-button">
                        Retry Analysis
                    </button>
                </div>
            )}
        </div>
    );
};

export default DataAnalysisPage;