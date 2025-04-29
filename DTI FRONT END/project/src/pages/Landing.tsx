import React from 'react';
import { Link } from 'react-router-dom';
import { Video, Wand2, Search } from 'lucide-react';

function Landing() {
  return (
    <div className="min-h-[calc(100vh-4rem)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <h1 className="text-4xl tracking-tight font-extrabold sm:text-5xl md:text-6xl">
            <span className="block text-white/90 backdrop-blur-sm">Transform Your Videos with</span>
            <span className="block text-blue-400 mt-2">AI-Powered Enhancement</span>
          </h1>
          <p className="mt-3 max-w-md mx-auto text-base text-gray-300/80 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
            Enhance your videos with cutting-edge AI technology. Convert black & white to color,
            increase resolution, and detect objects in real-time.
          </p>
        </div>

        <div className="mt-20">
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
            <div className="relative group">
              <div className="relative h-80 w-full overflow-hidden rounded-lg bg-white group-hover:opacity-75 sm:aspect-w-2 sm:aspect-h-1 sm:h-64 lg:aspect-w-1 lg:aspect-h-1">
                <img
                  src="https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80"
                  alt="Video Colorization"
                  className="h-full w-full object-cover object-center"
                />
                <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                  <Link
                    to="/process"
                    className="text-white text-xl font-semibold flex items-center space-x-2"
                  >
                    <Video className="h-8 w-8" />
                    <span>Colorize Videos</span>
                  </Link>
                </div>
              </div>
            </div>

            <div className="relative group">
              <div className="relative h-80 w-full overflow-hidden rounded-lg bg-white group-hover:opacity-75 sm:aspect-w-2 sm:aspect-h-1 sm:h-64 lg:aspect-w-1 lg:aspect-h-1">
                <img
                  src="https://images.unsplash.com/photo-1533279443086-d1c19a186416?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80"
                  alt="Resolution Enhancement"
                  className="h-full w-full object-cover object-center"
                />
                <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                  <Link
                    to="/enhance"
                    className="text-white text-xl font-semibold flex items-center space-x-2"
                  >
                    <Wand2 className="h-8 w-8" />
                    <span>Enhance Resolution</span>
                  </Link>
                </div>
              </div>
            </div>

            <div className="relative group">
              <div className="relative h-80 w-full overflow-hidden rounded-lg bg-white group-hover:opacity-75 sm:aspect-w-2 sm:aspect-h-1 sm:h-64 lg:aspect-w-1 lg:aspect-h-1">
                <img
                  src="https://images.unsplash.com/photo-1557264337-e8a93017fe92?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80"
                  alt="Object Detection"
                  className="h-full w-full object-cover object-center"
                />
                <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                  <Link
                    to="/detect"
                    className="text-white text-xl font-semibold flex items-center space-x-2"
                  >
                    <Search className="h-8 w-8" />
                    <span>Detect Objects</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Landing;