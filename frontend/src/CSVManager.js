import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const CSVManager = () => {
  const [files, setFiles] = useState([]);
  const [descriptions, setDescriptions] = useState({});

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.filter(file => file.type === 'text/csv' || file.name.endsWith('.csv'));
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
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
  };

  const updateDescription = (fileName, description) => {
    setDescriptions(prev => ({
      ...prev,
      [fileName]: description
    }));
  };

  const handleUpload = async () => {
    // Create a set of file paths
    const fileSet = new Set(files.map(file => file.name));
    
    // Prepare the initial state for the graph
    const initialState = {
      tables: "",
      analyzed_topics: "",
      csv_files: fileSet,
      topic: [],
      ScrapedArticles: new Set(),
      AnalyzedArticles: new Set(),
      ModelsPerTopic: new Set(),
      Relevance: new Set()
    };

    try {
      // Create FormData to send both the files and the initial state
      const formData = new FormData();
      
      // Add each file to FormData
      files.forEach(file => {
        formData.append('files', file);
      });
      
      // Add the initial state as JSON
      formData.append('initialState', JSON.stringify({
        ...initialState,
        csv_files: Array.from(initialState.csv_files) // Convert Set to Array for JSON
      }));

      // Send the request to the backend
      const response = await fetch('http://localhost:5000/upload-and-process', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      console.log('Processing result:', result);
      alert('Files uploaded and processed successfully!');
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files. Please try again.');
    }
  };

  return (
    <div className="csv-manager">
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

      {files.length > 0 && (
        <button className="upload-button" onClick={handleUpload}>
          Upload Files
        </button>
      )}
    </div>
  );
};

export default CSVManager; 