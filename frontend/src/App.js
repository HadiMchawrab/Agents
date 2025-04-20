import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import CSVManager from './CSVManager.js';
import DisplayTopics from './DisplayTopics.js';


function Layout({ children }) {
  return (
    <div >
      {children}
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={
          <Layout>
            <CSVManager/>
          </Layout>
        } />
        <Route path="/topics" element={
          <Layout>
            <DisplayTopics/>
          </Layout>
        } />  
        
      </Routes>
    </Router>
  );
}

export default App;