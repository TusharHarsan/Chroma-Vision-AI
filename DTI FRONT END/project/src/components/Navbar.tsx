import React from 'react';
import { Link } from 'react-router-dom';
import { Video, Wand2, Search, Home } from 'lucide-react';

function Navbar() {
  return (
    <nav className="bg-gray-900/50 backdrop-blur-lg border-b border-blue-900/20">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <Video className="h-8 w-8 text-blue-400" />
              <span className="text-xl font-bold text-white">ChromaVision</span>
            </Link>
          </div>
          <div className="flex space-x-8">
            <Link to="/" className="flex items-center text-gray-300 hover:text-blue-400 transition-colors">
              <Home className="h-5 w-5 mr-1" />
              <span>Home</span>
            </Link>
            <Link to="/process" className="flex items-center text-gray-300 hover:text-blue-400 transition-colors">
              <Video className="h-5 w-5 mr-1" />
              <span>Colorize</span>
            </Link>
            <Link to="/enhance" className="flex items-center text-gray-300 hover:text-blue-400 transition-colors">
              <Wand2 className="h-5 w-5 mr-1" />
              <span>Enhance</span>
            </Link>
            <Link to="/detect" className="flex items-center text-gray-300 hover:text-blue-400 transition-colors">
              <Search className="h-5 w-5 mr-1" />
              <span>Detect</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;