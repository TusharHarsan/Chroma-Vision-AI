import React, { useState } from 'react';
import { Upload, Play, RefreshCw, Search } from 'lucide-react';

function ObjectDetection() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setVideoFile(file);
  };

  const handleProcessing = () => {
    setProcessing(true);
    // Add actual processing logic here
    setTimeout(() => setProcessing(false), 2000);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-8">Object Detection</h2>
        
        <div className="space-y-8">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
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
              <Upload className="h-12 w-12 text-gray-400" />
              <span className="mt-2 text-sm text-gray-500">
                Drop your video here or click to upload
              </span>
              <span className="mt-1 text-xs text-gray-500">
                Supports: MP4, AVI, MOV (max 500MB)
              </span>
            </label>
          </div>

          {videoFile && (
            <div className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center space-x-4">
                  <Play className="h-6 w-6 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{videoFile.name}</p>
                    <p className="text-xs text-gray-500">
                      {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
              </div>

              <button
                onClick={handleProcessing}
                disabled={processing}
                className="w-full flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                {processing ? (
                  <>
                    <RefreshCw className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-5 w-5" />
                    Detect Objects
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ObjectDetection