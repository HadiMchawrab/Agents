import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';

function parseTables(tablesText) {
  const tables = [];
  const columnsByTable = {};

  const regex = /Table\s+(\w+):[\s\S]*?Has Columns:\s(.+?)(?=\n\n|$)/g;
  let match;

  while ((match = regex.exec(tablesText)) !== null) {
    const tableName = match[1].trim();
    const columns = match[2].split(',').map(c => c.trim());
    tables.push(tableName);
    columnsByTable[tableName] = columns;
  }

  return { tables, columnsByTable };
}

const CSVManager = ({ onProcessComplete }) => {
  const [files, setFiles] = useState([]);
  const [descriptions, setDescriptions] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.filter(file => file.type === 'text/csv' || file.name.endsWith('.csv'));
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
    setError(null); // Clear any previous errors when new files are added
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    }
  });

  const removeFile = (fileName) => {
    setFiles(files.filter(file => file.name !== fileName));
    const newDescriptions = { ...descriptions };
    delete newDescriptions[fileName];
    setDescriptions(newDescriptions);
    setError(null); // Clear error when files are removed
  };

  const updateDescription = (fileName, description) => {
    setDescriptions(prev => ({
      ...prev,
      [fileName]: description
    }));
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Create FormData and add files
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      // Add descriptions to the form data
      Object.entries(descriptions).forEach(([fileName, description]) => {
        formData.append(`descriptions[${fileName}]`, description);
      });

      // Send the request to the backend
      const response = await fetch('http://localhost:5000/upload-and-process', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail?.message || 'Failed to process files');
      }
      const backendData = result.result;
      const tablesText = backendData.tables;
      const { tables, columnsByTable } = parseTables(tablesText);

      if (!Array.isArray(backendData.analyzed_topics)) {
      throw new Error("Invalid backend response: analyzed_topics is not an array.");
      }

      const transformedResult = {
        topics: backendData.analyzed_topics.map((topicObj, index) => {
          const topic = topicObj.topic;
          return {
            topic,
            Relationship: new Set(backendData.Relationship?.[topic] || []),
            Explanation: new Set(backendData.Explanation?.[topic] || []),
            ML_Models1: new Set((backendData.ML_Models1?.[index]?.split(",") || []).map(m => m.trim())),
            ModelsPerTopic: new Set((backendData.ModelsPerTopic?.[topic]?.split(",") || []).map(m => m.trim()))
          };
        }),
        tables,
        columnsByTable
      };


      onProcessComplete(transformedResult);
      
      navigate('/results');
    } catch (error) {
      console.error('Error processing files:', error);
      setError(error.message);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`csv-manager ${isLoading ? 'loading' : ''}`}>
      <h2>New Process</h2>
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the CSV files here...</p>
        ) : (
          <p>Drag and drop CSV files here, or click to select files</p>
        )}
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="files-list">
        {files.map((file) => (
          <div key={file.name} className="file-item">
            <div className="file-info">
              <span>{file.name}</span>
              <button onClick={() => removeFile(file.name)}>Remove</button>
            </div>
            <textarea
              placeholder="Add description for this table..."
              value={descriptions[file.name] || ''}
              onChange={(e) => updateDescription(file.name, e.target.value)}
              rows={3}
            />
          </div>
        ))}
      </div>

      <button 
        className="generate-button" 
        onClick={handleGenerate}
        disabled={files.length === 0 || isLoading}
      >
        {isLoading ? 'Processing...' : 'Process Files'}
      </button>
    </div>
  );
};

export default CSVManager; 