import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import './TopicDetailsPage.css';


const TopicDetailsPage = () => {
  const location = useLocation();
  const topic = location.state?.topic;
  const tables = location.state?.tables;
  const columnsByTable = location.state?.columnsByTable;
  
  console.log('TopicDetailsPage - columnsByTable:', columnsByTable);
  console.log('TopicDetailsPage - tables:', tables);
  
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

      <h3 className="selection-title">Choose the tables and columns to work on</h3>
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
                <label htmlFor={`column-select-${row.id}`}>Select Columns:</label>
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
    </div>
  );
};

export default TopicDetailsPage; 