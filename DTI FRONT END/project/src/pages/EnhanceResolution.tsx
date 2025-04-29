import React, { useState } from 'react';
import { Upload, Play, RefreshCw, ZoomIn, Download, Undo2, Share2, ZoomOut } from 'lucide-react';

function EnhanceResolution() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processed, setProcessed] = useState(false);
  const [scale, setScale] = useState<'2x' | '4x'>('2x');
  const maxScale = '4x'; // Setting the maximum scale here

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setProcessed(false);
    }
  };

  const handleProcessing = () => {
    setProcessing(true);
    // Add actual processing logic here
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
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-8">Resolution Enhancement</h2>
        
        {!processed ? (
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

                <div className="flex items-center space-x-4">
                  <span className="text-sm font-medium text-gray-700">Scale Factor:</span>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setScale('2x')}
                      className={`px-4 py-2 rounded-md ${
                        scale === '2x'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      2x
                    </button>
                    <button
                      onClick={() => setScale('4x')}
                      className={`px-4 py-2 rounded-md ${
                        scale === '4x'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                      disabled={scale === maxScale}
                    >
                      4x
                    </button>
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
                      <ZoomIn className="mr-2 h-5 w-5" />
                      Enhance Resolution ({scale})
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-8">
            <div className="relative aspect-video rounded-lg overflow-hidden bg-black">
              <video
                className="w-full h-full object-contain"
                controls
                src="data:video/mp4;base64,..."
              >
                Your browser does not support the video tag.
              </video>
              <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded-full text-sm">
                {scale} Enhanced
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900">Original</h3>
                  <div className="flex items-center text-sm text-gray-500">
                    <ZoomOut className="h-4 w-4 mr-1" />
                    480p
                  </div>
                </div>
                <div className="aspect-video bg-gray-200 rounded-lg"></div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900">Enhanced</h3>
                  <div className="flex items-center text-sm text-gray-500">
                    <ZoomIn className="h-4 w-4 mr-1" />
                    {scale === '2x' ? '960p' : '1920p'}
                  </div>
                </div>
                <div className="aspect-video bg-gray-200 rounded-lg"></div>
              </div>
            </div>

            <div className="flex flex-col space-y-4">
              <div className="flex space-x-4">
                <button className="flex-1 flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <Download className="h-5 w-5 mr-2" />
                  Download Enhanced Video
                </button>
                <button className="flex-1 flex items-center justify-center px-4 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <Share2 className="h-5 w-5 mr-2" />
                  Share Result
                </button>
              </div>
              <button
                onClick={handleReset}
                className="flex items-center justify-center px-4 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
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

export default EnhanceResolution;