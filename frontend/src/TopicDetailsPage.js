import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './TopicDetailsPage.css';

const TopicDetailsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  console.log('TopicDetailsPage - Full location state:', location.state);
  
  const topic = location.state?.topic;
  const allTables = location.state?.tables || [];
  
  // Get tables from GPT_Columns for current topic
  const getGptTablesAndColumns = () => {
    const gptData = topic?.GPT_Columns?.[0]?.[0] || {};
    return gptData;
  };
  
  // Check if a table has any columns not used by GPT
  const hasAvailableColumns = (tableName) => {
    const gptData = getGptTablesAndColumns();
    const gptColumns = gptData[tableName] || [];
    
    // Get all columns for this table
    const tableObj = allTables.find(obj => Object.keys(obj)[0] === tableName);
    const allColumns = tableObj ? tableObj[tableName] : [];
    
    // Return true if there are any columns not in GPT's selection
    return allColumns.some(column => !gptColumns.includes(column));
  };
  
  // Get all available tables that have unused columns
  const getAvailableTablesFromAll = () => {
    // Get all available tables from the initial data
    const availableTables = allTables.map(tableObj => Object.keys(tableObj)[0]);
    console.log('All available tables:', availableTables);
    
    // Filter tables - keep only those that have available columns
    const filteredTables = availableTables.filter(table => hasAvailableColumns(table));
    console.log('Filtered tables (with available columns):', filteredTables);
    
    return filteredTables;
  };

  // Get available columns that aren't in GPT_Columns for the selected table
  const getAvailableColumnsForTable = (tableName) => {
    const gptData = getGptTablesAndColumns();
    const gptColumns = gptData[tableName] || [];
    console.log('GPT suggested columns for', tableName, ':', gptColumns);
    
    // Get all columns for this table
    const tableObj = allTables.find(obj => Object.keys(obj)[0] === tableName);
    const allColumns = tableObj ? tableObj[tableName] : [];
    console.log('All available columns for', tableName, ':', allColumns);
    
    // Filter out columns that are already in GPT_Columns
    const filteredColumns = allColumns.filter(column => !gptColumns.includes(column));
    console.log('Filtered columns for', tableName, ':', filteredColumns);
    
    return filteredColumns;
  };
  
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
    const availableTables = getAvailableTablesFromAll();
    const selectedTables = dropdownRows
      .filter(row => row.id !== currentRowId)
      .map(row => row.selectedTable);
    return availableTables.filter(table => !selectedTables.includes(table));
  };

  const handleContinue = async () => {
    // Get GPT-selected columns
    const gptData = getGptTablesAndColumns();
    
    // Create merged tables object
    const mergedTables = {};
    
    // First add GPT-selected columns
    Object.entries(gptData).forEach(([table, columns]) => {
      mergedTables[table] = [...columns];
    });
    
    // Then merge user-selected columns
    dropdownRows.forEach(row => {
      if (row.selectedTable && row.selectedColumns.length > 0) {
        if (mergedTables[row.selectedTable]) {
          // Add to existing table's columns
          mergedTables[row.selectedTable] = [
            ...mergedTables[row.selectedTable],
            ...row.selectedColumns
          ];
        } else {
          // Create new table entry
          mergedTables[row.selectedTable] = [...row.selectedColumns];
        }
      }
    });

    // Data for backend - ensure data matches Pydantic model
    const submissionData = {
      topic: topic.topic,
      Relationship: Array.from(topic.Relationship || []),
      ML_Models: Array.from(topic.ML_Models || []),
      tables: mergedTables
    };

    // Navigate to analysis page immediately
    navigate('/analysis', { 
      state: { 
        topic: topic,
        tables: mergedTables,
        submissionData: submissionData // Pass the submission data to be used by AnalysisPage
      }
    });

    // Start the submission in the background
    fetch('http://localhost:5000/submit-data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(submissionData)
    }).catch(error => {
      console.error('Error submitting data:', error);
    });
  };

  if (!topic) {
    return <div className="topic-details-page">No topic selected</div>;
  }

  const availableTables = getAvailableTablesFromAll();

  return (
    <div className="topic-details-page">
      <h2>{topic.topic}</h2>
      
      <div className="topic-info">
        <div className="topic-info-section">
          <span className="topic-label">Reasoning:</span>
          <span className="topic-content">
            {topic.reasoning}
          </span>
        </div>

        <div className="topic-info-section">
          <span className="topic-label">Needs:</span>
          <span className="topic-content">
            {topic.Needs}
          </span>
        </div>

        <div className="topic-info-section">
          <span className="topic-label">Relationship:</span>
          <span className="topic-content">
            {topic.Relationship && Array.from(topic.Relationship).join('. ')}
          </span>
        </div>

        <div className="topic-info-section">
          <span className="topic-label">ML Models:</span>
          <span className="topic-content">
            {topic.ML_Models && Array.from(topic.ML_Models).join(', ')}
          </span>
        </div>

        <div className="topic-info-section">
          <span className="topic-label">GPT Selected Columns:</span>
          <span className="topic-content">
            {topic.GPT_Columns && topic.GPT_Columns.length > 0 ? (
              Object.entries(topic.GPT_Columns[0][0]).map(([table, columns]) => (
                <div key={table}>
                  <strong>{table}:</strong> {columns.join(', ')}
                </div>
              ))
            ) : (
              'None'
            )}
          </span>
        </div>
        </div>

      <h3 className="selection-title">Choose Additional Tables and Columns for Analysis</h3>
      <div className="selection-container">
        {dropdownRows.map((row, index) => (
          <div key={row.id} className="selection-row">
            {index === dropdownRows.length - 1 && (
              <button 
                className="add-row-button"
                onClick={handleAddRow}
                disabled={dropdownRows.length >= availableTables.length}
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
                <label htmlFor={`table-select-${row.id}`}>Select Additional Table:</label>
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
                <label htmlFor={`column-select-${row.id}`}>Select Additional Columns:</label>
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
                  {row.selectedTable && getAvailableColumnsForTable(row.selectedTable).map((column, index) => (
                    <option key={index} value={column}>{column}</option>
                  ))}
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