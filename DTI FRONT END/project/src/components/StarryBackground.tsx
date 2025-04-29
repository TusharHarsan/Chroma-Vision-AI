import React from 'react';

function StarryBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900 via-gray-900 to-black"></div>
      <div className="stars absolute inset-0"></div>
      <div className="stars2 absolute inset-0"></div>
      <div className="stars3 absolute inset-0"></div>
      <div className="stars4 absolute inset-0"></div>
      <div className="stars5 absolute inset-0"></div>
    </div>
  );
}

export default StarryBackground;