import React from 'react';
import { useNavigate } from 'react-router-dom';
import './ResultsPage.css';

const ResultsPage = ({ topics }) => {
  const navigate = useNavigate();

  const handleTopicClick = (topic) => {
    navigate('/topic-details', { state: { topic } });
  };

  return (
    <div className="results-page">
      <h2>Select a Topic</h2>
      <div className="topics-grid">
        {topics.map((topic, index) => (
          <div 
            key={index} 
            className="topic-box"
            onClick={() => handleTopicClick(topic)}
          >
            <h3>{topic.topic}</h3>
            
            <div className="topic-section">
              <h4>Relationships</h4>
              <ul className="topic-list">
                {topic.Relationship && Array.from(topic.Relationship).map((rel, i) => (
                  <li key={i}>{rel}</li>
                ))}
              </ul>
            </div>

            <div className="topic-section">
              <h4>Explanations</h4>
              <ul className="topic-list">
                {topic.Explanation && Array.from(topic.Explanation).map((exp, i) => (
                  <li key={i}>{exp}</li>
                ))}
              </ul>
            </div>

            <div className="topic-section">
              <h4>ML Models</h4>
              <ul className="topic-list">
                {topic.ML_Models1 && Array.from(topic.ML_Models1).map((model, i) => (
                  <li key={i}>{model}</li>
                ))}
              </ul>
            </div>

            <div className="topic-section">
              <h4>Models Per Topic</h4>
              <ul className="topic-list">
                {topic.ModelsPerTopic && Array.from(topic.ModelsPerTopic).map((model, i) => (
                  <li key={i}>{model}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsPage; 