import React from 'react';
import './App.css';
import CSVManager from './CSVManager';
import './CSVManager.css';

function App() {
  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-content">
          <div className="logo">AI Consultant</div>
          <div className="nav-links">
            <a href="#" className="nav-link">Dashboard</a>
            <a href="#" className="nav-link">Help</a>
            <a href="#" className="nav-link">Settings</a>
          </div>
        </div>
      </nav>
      <main>
        <CSVManager />
      </main>
    </div>
  );
}

export default App;
