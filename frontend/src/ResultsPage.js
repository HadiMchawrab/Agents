import React from 'react';
import { useNavigate } from 'react-router-dom';
import './ResultsPage.css';

const ResultsPage = ({ topics, tables }) => {
  const navigate = useNavigate();

  const handleTopicClick = (topic) => {
    console.log('ResultsPage - Clicked topic:', topic.topic);
    console.log('ResultsPage - GPT_Columns for topic:', topic.GPT_Columns);
    
    navigate('/topic-details', { 
      state: { 
        topic: {
          topic: topic.topic,
          reasoning: topic.reasoning,
          Relationship: topic.Relationship,
          ML_Models: topic.ML_Models,
          Needs: topic.Needs,
          GPT_Columns: topic.GPT_Columns
        },
        tables
      } 
    });
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
              <h4>Reasoning</h4>
              <ul className="topic-list">
                <li>{topic.reasoning}</li>
              </ul>
            </div>
            
            <div className="topic-section">
              <h4>Relationship</h4>
              <ul className="topic-list">
                {topic.Relationship && Array.from(topic.Relationship).map((rel, i) => (
                  <li key={i}>{rel}</li>
                ))}
              </ul>
            </div>

            <div className="topic-section">
              <h4>Needs</h4>
              <ul className="topic-list">
                {topic.Needs && Array.from(topic.Needs).map((exp, i) => (
                  <li key={i}>{exp}</li>
                ))}
              </ul>
            </div>

            <div className="topic-section">
              <h4>ML Models</h4>
              <ul className="topic-list">
                {topic.ML_Models && Array.from(topic.ML_Models).map((model, i) => (
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