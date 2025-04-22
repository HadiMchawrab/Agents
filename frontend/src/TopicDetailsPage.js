import React, { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './TopicDetailsPage.css';

const TopicDetailsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  console.log('TopicDetailsPage - Full location state:', location.state);
  
  const topic = location.state?.topic;
  const tables = location.state?.tables;
  const columnsByTable = location.state?.columnsByTable;
  const analyzedArticles = location.state?.analyzedArticles;
  const scrapedArticles = location.state?.scrapedArticles;
  const relationships = location.state?.relationships;
  const explanations = location.state?.explanations;
  const modelsPerTopic = location.state?.modelsPerTopic;
  const mlModels = location.state?.mlModels;
  const allTopics = location.state?.allTopics;
  
  console.log('TopicDetailsPage - topic:', topic);
  console.log('TopicDetailsPage - tables:', tables);
  console.log('TopicDetailsPage - columnsByTable:', columnsByTable);
  console.log('TopicDetailsPage - analyzedArticles:', analyzedArticles);
  console.log('TopicDetailsPage - scrapedArticles:', scrapedArticles);
  console.log('TopicDetailsPage - relationships:', relationships);
  console.log('TopicDetailsPage - explanations:', explanations);
  console.log('TopicDetailsPage - modelsPerTopic:', modelsPerTopic);
  console.log('TopicDetailsPage - mlModels:', mlModels);
  console.log('TopicDetailsPage - allTopics:', allTopics);
  
  const [dropdownRows, setDropdownRows] = useState([
    { id: 1, selectedTable: '', selectedColumns: [] }
  ]);

  const handleAddRow = () => {
    const newRow = {
      id: Date.now(),
      selectedTable: '',
      selectedColumns: []
    };
    setDropdownRows([...dropdownRows, newRow]);
  };

  const handleRemoveRow = (rowId) => {
    setDropdownRows(dropdownRows.filter(row => row.id !== rowId));
  };

  const handleTableChange = (rowId, table) => {
    setDropdownRows(dropdownRows.map(row => {
      if (row.id === rowId) {
        return { ...row, selectedTable: table, selectedColumns: [] };
      }
      return row;
    }));
  };

  const handleColumnChange = (rowId, selectedOptions) => {
    setDropdownRows(dropdownRows.map(row => {
      if (row.id === rowId) {
        return { ...row, selectedColumns: selectedOptions };
      }
      return row;
    }));
  };

  const getAvailableTables = (currentRowId) => {
    const selectedTables = dropdownRows
      .filter(row => row.id !== currentRowId)
      .map(row => row.selectedTable);
    return tables.filter(table => !selectedTables.includes(table));
  };

  const handleContinue = async () => {
    console.log('Location state:', location.state);
    console.log('All topics:', allTopics);
    
    if (!allTopics) {
      console.error('allTopics is undefined');
      return;
    }

    // Prepare the data for submission
    const selectedData = dropdownRows.map(row => ({
      table: row.selectedTable,
      columns: row.selectedColumns
    }));

    // Create the complete submission data
    const submissionData = {
      // All topics data (as received from results page)
      topics: allTopics.map(topic => ({
        topic: topic.topic,
        relationships: Array.from(topic.Relationship || []),
        explanations: Array.from(topic.Explanation || []),
        ml_models: Array.from(topic.ML_Models1 || []),
        models_per_topic: Array.from(topic.ModelsPerTopic || [])
      })),
      
      // Original tables and columns data
      tables: tables,
      columns_by_table: columnsByTable,
      
      // Additional data from backend
      analyzed_articles: analyzedArticles,
      scraped_articles: scrapedArticles,
      relationships: relationships,
      explanations: explanations,
      models_per_topic: modelsPerTopic,
      ml_models: mlModels,
      
      // User selections
      selected_topic: topic.topic,
      selected_tables_and_columns: selectedData
    };

    try {
      const response = await fetch('http://localhost:5000/submit-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(submissionData)
      });

      if (!response.ok) {
        throw new Error('Failed to submit data');
      }

      const result = await response.json();
      
      // Navigate to analysis page with both the original and new data
      navigate('/analysis', { 
        state: { 
          topic,
          selectedData,
          analysisResult: result // If the backend returns any analysis results
        }
      });
    } catch (error) {
      console.error('Error submitting data:', error);
      // TODO: Add error handling UI
    }
  };

  if (!topic) {
    return <div className="topic-details-page">No topic selected</div>;
  }

  return (
    <div className="topic-details-page">
      <h2>{topic.topic}</h2>
      
      <div className="topic-info">
        <div className="topic-info-section">
          <span className="topic-label">Relationships:</span>
          <span className="topic-content">
            {topic.Relationship && Array.from(topic.Relationship).join('. ')}
          </span>
        </div>

        <div className="topic-info-section">
          <span className="topic-label">Explanations:</span>
          <span className="topic-content">
            {topic.Explanation && Array.from(topic.Explanation).join('. ')}
          </span>
        </div>
      </div>

      <h3 className="selection-title">Choose the tables you want to add to the model analysis</h3>
      <div className="selection-container">
        {dropdownRows.map((row, index) => (
          <div key={row.id} className="selection-row">
            {index === dropdownRows.length - 1 && (
              <button 
                className="add-row-button"
                onClick={handleAddRow}
                disabled={dropdownRows.length >= tables.length}
              >
                +
              </button>
            )}
            {dropdownRows.length > 1 && (
              <button 
                className="remove-row-button"
                onClick={() => handleRemoveRow(row.id)}
              >
                −
              </button>
            )}
            <div className="selection-section">
              <div className="dropdown-group">
                <label htmlFor={`table-select-${row.id}`}>Select Table:</label>
                <select 
                  id={`table-select-${row.id}`}
                  value={row.selectedTable}
                  onChange={(e) => handleTableChange(row.id, e.target.value)}
                >
                  <option value="">-- Select a table --</option>
                  {getAvailableTables(row.id).map((table, index) => (
                    <option key={index} value={table}>{table}</option>
                  ))}
                </select>
              </div>

              <div className="dropdown-group">
                <label htmlFor={`column-select-${row.id}`}>Select Columns (Optional):</label>
                <select 
                  id={`column-select-${row.id}`}
                  multiple 
                  value={row.selectedColumns}
                  onChange={(e) => {
                    const options = Array.from(e.target.selectedOptions, option => option.value);
                    handleColumnChange(row.id, options);
                  }}
                  disabled={!row.selectedTable}
                >
                  {row.selectedTable && columnsByTable && columnsByTable[row.selectedTable] && 
                    columnsByTable[row.selectedTable].map((column, index) => (
                      <option key={index} value={column}>{column}</option>
                    ))
                  }
                </select>
                <p className="column-hint">Hold Ctrl (Windows) or ⌘ (Mac) to select multiple columns</p>
                {row.selectedColumns.length > 0 && (
                  <div className="selected-columns-info">
                    {row.selectedColumns.length} column{row.selectedColumns.length !== 1 ? 's' : ''} selected
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="continue-section">
        <button 
          className="continue-button"
          onClick={handleContinue}
        >
          Continue to Analysis
        </button>
      </div>
    </div>
  );
};

export default TopicDetailsPage; 