import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';

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

      // Transform the backend response into the format expected by ResultsPage
      const transformedResult = {
        topics: result.analyzed_topics.map(topicSet => {
          // Find the corresponding topic data from the result
          const topicData = result.topic.find(t => t.topic === Array.from(topicSet)[0]);
          return {
            topic: Array.from(topicSet)[0],
            Relationship: new Set(topicData?.Relationship || []),
            Explanation: new Set(topicData?.Explanation || []),
            ML_Models1: new Set(topicData?.ML_Models1 || []),
            ModelsPerTopic: new Set(topicData?.ModelsPerTopic || [])
          };
        })
      };

      // Call the onProcessComplete callback with the transformed result
      onProcessComplete(transformedResult);
      
      // Navigate to the results page
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