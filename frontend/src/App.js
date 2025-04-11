import React from 'react';
import './App.css';
import CSVManager from './CSVManager';
import './CSVManager.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>CSV File Manager</h1>
      </header>
      <main>
        <CSVManager />
      </main>
    </div>
  );
}

export default App;
