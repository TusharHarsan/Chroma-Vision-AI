import React, { useState } from 'react';
import { Upload, Play, RefreshCw, Download, Undo2, Share2 } from 'lucide-react';

function VideoProcessor() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processed, setProcessed] = useState(false);
  const [colorIntensity, setColorIntensity] = useState<number>(50);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setProcessed(false);
    }
  };

  const handleProcessing = () => {
    setProcessing(true);
    setTimeout(() => {
      setProcessing(false);
      setProcessed(true);
    }, 2000);
  };

  const handleReset = () => {
    setVideoFile(null);
    setProcessed(false);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="bg-gray-900/40 backdrop-blur-xl rounded-2xl shadow-2xl border border-blue-900/20 p-8">
        <h2 className="text-3xl font-bold text-white mb-8">Video Colorization</h2>
        
        {!processed ? (
          <div className="space-y-8">
            <div className="border-2 border-dashed border-blue-900/50 rounded-lg p-12 text-center hover:border-blue-400/50 transition-colors">
              <input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="hidden"
                id="video-upload"
              />
              <label
                htmlFor="video-upload"
                className="cursor-pointer flex flex-col items-center"
              >
                <Upload className="h-12 w-12 text-blue-400" />
                <span className="mt-2 text-sm text-gray-300">
                  Drop your video here or click to upload
                </span>
                <span className="mt-1 text-xs text-gray-400">
                  Supports: MP4, AVI, MOV (max 500MB)
                </span>
              </label>
            </div>

            {videoFile && (
              <div className="space-y-4">
                <div className="bg-dark-blue-800/50 p-4 rounded-lg border border-blue-900/20">
                  <div className="flex items-center space-x-4">
                    <Play className="h-6 w-6 text-blue-400" />
                    <div>
                      <p className="text-sm font-medium text-white">{videoFile.name}</p>
                      <p className="text-xs text-gray-400">
                        {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <label htmlFor="colorIntensity" className="text-sm font-medium text-gray-300">
                    Color Intensity:
                  </label>
                  <input
                    type="range"
                    id="colorIntensity"
                    min="0"
                    max="100"
                    value={colorIntensity}
                    onChange={(e) => setColorIntensity(parseInt(e.target.value))}
                    className="w-1/2"
                  />
                  <span className="text-sm text-gray-400">{colorIntensity}%</span>
                </div>

                <button
                  onClick={handleProcessing}
                  disabled={processing}
                  className="w-full flex items-center justify-center px-4 py-3 rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors disabled:opacity-50"
                >
                  {processing ? (
                    <>
                      <RefreshCw className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
                      Processing...
                    </>
                  ) : (
                    'Start Colorization'
                  )}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-8">
            <div className="relative aspect-video rounded-lg overflow-hidden bg-dark-blue-900 border border-blue-900/20">
              <video
                className="w-full h-full object-contain"
                controls
                src="data:video/mp4;base64,..."
              >
                Your browser does not support the video tag.
              </video>
              <div className="absolute top-4 right-4 bg-blue-600/90 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm">
                Colorized
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-dark-blue-800/50 p-4 rounded-lg border border-blue-900/20">
                <h3 className="font-medium text-white mb-2">Original</h3>
                <div className="aspect-video bg-dark-blue-900 rounded-lg"></div>
              </div>
              <div className="bg-dark-blue-800/50 p-4 rounded-lg border border-blue-900/20">
                <h3 className="font-medium text-white mb-2">Colorized</h3>
                <div className="aspect-video bg-dark-blue-900 rounded-lg"></div>
              </div>
            </div>

            <div className="flex flex-col space-y-4">
              <div className="flex space-x-4">
                <button className="flex-1 flex items-center justify-center px-4 py-3 rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors">
                  <Download className="h-5 w-5 mr-2" />
                  Download Video
                </button>
                <button className="flex-1 flex items-center justify-center px-4 py-3 rounded-md text-gray-300 bg-dark-blue-800/50 hover:bg-dark-blue-700/50 border border-blue-900/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors">
                  <Share2 className="h-5 w-5 mr-2" />
                  Share Result
                </button>
              </div>
              <button
                onClick={handleReset}
                className="flex items-center justify-center px-4 py-3 rounded-md text-gray-300 bg-dark-blue-800/50 hover:bg-dark-blue-700/50 border border-blue-900/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
              >
                <Undo2 className="h-5 w-5 mr-2" />
                Process Another Video
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoProcessor;