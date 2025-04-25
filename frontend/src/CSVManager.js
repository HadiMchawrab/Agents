import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './CSVManager.css'; 
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

<<<<<<< HEAD
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      console.log('Processing result:', result);
      alert('Files uploaded and processed successfully!');
=======
      const response_result = await response.json();
      console.log('Backend response:', response_result);

      if (!response.ok) {
        throw new Error(response_result.detail?.message || 'Failed to process files');
      }
      const backendData = response_result.result;
      console.log('Full backend response:', backendData);
      console.log('Topics from backend:', backendData.topic);
      console.log('ML Models from backend:', backendData.ML_Models1);
      console.log('Models per topic:', backendData.ModelsPerTopic);
      console.log('GPT Columns structure:', backendData.GPT_Columns);

      if (!Array.isArray(backendData.topic)) {
        throw new Error("Invalid backend response: topic array is missing");
      }

      const transformedResult = {
        topics: backendData.topic.map((topicName, index) => {
          console.log('Processing topic:', topicName);
          return {
            topic: topicName,
            reasoning: (backendData.analyzed_topics?.[index]?.reasoning || ""),
            GPT_Columns: backendData.GPT_Columns?.[topicName] || [], // Keep the nested array structure
            Needs: new Set(backendData.Needs?.[topicName] || []),
            Relationship: new Set(backendData.Relationship?.[topicName] || []),
            ML_Models: new Set([
              ...(backendData.ML_Models1?.[index]?.split(",") || []).map(m => m.trim()),
              ...(backendData.ModelsPerTopic?.[topicName]?.split(",") || []).map(m => m.trim())
            ])
          };
        }),
        tables: backendData.tables || []
      };

      console.log('Transformed result:', transformedResult);
      onProcessComplete(transformedResult);
      
      navigate('/results');
>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565
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