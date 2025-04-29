import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import VideoProcessor from './pages/VideoProcessor';
import EnhanceResolution from './pages/EnhanceResolution';
import ObjectDetection from './pages/ObjectDetection';
import StarryBackground from './components/StarryBackground';

function App() {
  return (
    <div className="min-h-screen bg-transparent relative">
      <StarryBackground />
      <div className="relative z-10">
        <Navbar />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/process" element={<VideoProcessor />} />
          <Route path="/enhance" element={<EnhanceResolution />} />
          <Route path="/detect" element={<ObjectDetection />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;