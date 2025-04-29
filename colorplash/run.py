from app import app

if __name__ == '__main__':
    print("Starting ChromaVision in production mode (no auto-reload)...")
    app.run(debug=False, host='0.0.0.0', port=5000) 