import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const CSVManager = () => {
  const [files, setFiles] = useState([]);
  const [descriptions, setDescriptions] = useState({});
  const [isLoading, setIsLoading] = useState(false);

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

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      // Create a set of file paths with the correct directory
      const fileSet = new Set(files.map(file => `csv_test/${file.name}`));
      
      // Prepare the initial state for the graph
      const initialState = {
        tables: "",
        analyzed_topics: [],
        csv_files: fileSet,
        topic: [],
        ScrapedArticles: new Set(),
        AnalyzedArticles: new Set(),
        ModelsPerTopic: new Set(),
        Relationship: new Set(),
        Explanation: new Set(),
        ML_Models1: new Set()
      };

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
      alert('Graph generated successfully!');
    } catch (error) {
      console.error('Error generating graph:', error);
      alert('Error generating graph. Please try again.', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`csv-manager ${isLoading ? 'loading' : ''}`}>
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

      <button 
        className="generate-button" 
        onClick={handleGenerate}
        disabled={files.length === 0 || isLoading}
      >
        {isLoading ? 'Generating...' : 'Generate'}
      </button>
    </div>
  );
};

export default CSVManager; 