import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './DataAnalysisPage.css';

const DataAnalysisPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { analysisResult, tables, images_bytes } = location.state || {};
    
    console.log('DataAnalysisPage received state:', location.state);
    console.log('Analysis result:', analysisResult);
    console.log('Images:', images_bytes);

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

            {/* Model Analysis Results */}
            {analysisResult.additionalData && (
                <div className="model-analysis">
                    <h2>Model Analysis</h2>
                    <div className="model-info">
                        <h3>Chosen Model</h3>
                        <p>{analysisResult.additionalData.chosen_models}</p>
                    </div>
                    <div className="model-info">
                        <h3>Model Analysis</h3>
                        <p>{analysisResult.additionalData.explained_models}</p>
                    </div>
                    <div className="model-info">
                        <h3>Training Script</h3>
                        <pre>{analysisResult.additionalData.final_scripts}</pre>
                    </div>
                </div>
            )}

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

            {/* Visualization Results */}
            {images_bytes && Object.keys(images_bytes).length > 0 ? (
                Object.entries(images_bytes).map(([tableName, imageStrings]) => (
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
                                            console.error(`Failed to load image for ${tableName}`);
                                            e.target.style.display = 'none';
                                        }}
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                ))
            ) : (
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