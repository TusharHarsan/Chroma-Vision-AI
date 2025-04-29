import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
from werkzeug.utils import secure_filename
from colorizer_utils import colorize_image
from video_colorizer import colorize_video
import threading
import time
import sys
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'models/ColorizeArtistic_gen.pth'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size
app.secret_key = 'some_secret_key'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session timeout in seconds

# Track processing status (use a global dictionary with lock for thread safety)
processing_tasks = {}
task_lock = threading.Lock()

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

# Background processing function
def process_video_background(task_id, filepath, output_path, render_factor, fps_value):
    try:
        with task_lock:
            processing_tasks[task_id] = {
                'status': 'processing',
                'message': 'Video colorization in progress...',
                'progress': 0,
                'timestamp': time.time()
            }
        
        print(f"Starting background video colorization for task {task_id}")
        
        # Ensure NumPy compatibility before starting
        try:
            import numpy as np
            version = [int(x) for x in np.__version__.split('.')]
            if version[0] > 1:
                print("Warning: NumPy version 2+ detected. Using v1.26.4 for compatibility...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"])
        except Exception as e:
            print(f"Warning during NumPy check: {e}")
        
        # Set thread name to include task_id for progress tracking
        threading.current_thread().name = f"task_{task_id}"
            
        # Now colorize the video with progress tracking
        colorize_video(
            source_path=filepath, 
            output_path=output_path, 
            render_factor=int(render_factor),
            fps=fps_value,
            update_progress_callback=update_task_progress
        )
        
        # Update status to completed
        with task_lock:
            processing_tasks[task_id] = {
                'status': 'completed',
                'message': 'Video colorization completed',
                'output_filename': os.path.basename(output_path),
                'progress': 100,
                'timestamp': time.time()
            }
        print(f"Completed background video colorization for task {task_id}")
    except Exception as e:
        import traceback
        print(f"Error in background video processing: {str(e)}")
        print(traceback.format_exc())
        
        # Update status to error
        with task_lock:
            processing_tasks[task_id] = {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'progress': 0,
                'timestamp': time.time()
            }

# Update task progress periodically 
def update_task_progress(task_id, progress):
    with task_lock:
        if task_id in processing_tasks:
            processing_tasks[task_id]['progress'] = progress
            processing_tasks[task_id]['timestamp'] = time.time()

@app.route('/', methods=['GET', 'POST'])
def index():
    if not os.path.exists(app.config['MODEL_PATH']):
        flash('Model file not found. Please download the ColorizeArtistic_gen.pth model file and place it in the models directory.')
        return render_template('index.html', model_missing=True)
    
    # Check for video parameter (redirect from completed processing)
    video_filename = request.args.get('video')
    if video_filename:
        input_filename = video_filename.replace('colorized_', '')
        return render_template('index.html',
                             input_video=input_filename,
                             output_video=video_filename,
                             is_video=True)
        
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get render factor from form or use default
            render_factor = int(request.form.get('render_factor', 35))
            
            # Process differently based on file type (image or video)
            if is_video(filename):
                # For video files - process in background
                try:
                    # Create a unique task ID
                    task_id = f"task_{int(time.time())}"
                    output_filename = f"colorized_{filename}"
                    output_path = os.path.join('static/results', output_filename)
                    
                    # Store task_id in session
                    session['current_task_id'] = task_id
                    
                    # Get fps from form or use None (original fps)
                    fps_value = request.form.get('fps', None)
                    if fps_value:
                        try:
                            fps_value = float(fps_value)
                        except ValueError:
                            fps_value = None
                    
                    # Start a background thread for processing
                    thread = threading.Thread(
                        target=process_video_background,
                        args=(task_id, filepath, output_path, render_factor, fps_value)
                    )
                    thread.daemon = True
                    thread.start()
                    
                    # Redirect to the processing status page
                    return render_template('index.html', 
                                         processing=True,
                                         task_id=task_id,
                                         input_video=filename)
                except Exception as e:
                    import traceback
                    print(f"Error setting up video processing: {str(e)}")
                    print(traceback.format_exc())
                    flash(f'Error processing video: {str(e)}')
                    return redirect(request.url)
            else:
                # For image files - process immediately
                try:
                    # Now colorize_image returns both the grayscale path and colorized path
                    grayscale_path, colorized_path = colorize_image(filepath, app.config['UPLOAD_FOLDER'], render_factor=render_factor)
                    
                    # Extract just the filenames from the paths
                    grayscale_filename = os.path.basename(grayscale_path)
                    colorized_filename = os.path.basename(colorized_path)
                    
                    return render_template('index.html', 
                                         input_image=filename, 
                                         grayscale_image=grayscale_filename,
                                         output_image=colorized_filename,
                                         is_video=False)
                except Exception as e:
                    import traceback
                    print(f"Error processing image: {str(e)}")
                    print(traceback.format_exc())
                    flash(f'Error processing image: {str(e)}')
                    return redirect(request.url)
        else:
            flash('File type not allowed. Please upload an image (.png, .jpg, .jpeg, .bmp) or video (.mp4, .avi, .mov, .mkv).')
            return redirect(request.url)
            
    return render_template('index.html', is_video=False)

@app.route('/check_status/<task_id>')
def check_status(task_id):
    """API endpoint to check the status of a processing task"""
    with task_lock:
        if task_id in processing_tasks:
            # Return a copy of the task status
            return jsonify(dict(processing_tasks[task_id]))
        
        # Try to retrieve task ID from session if not found
        if 'current_task_id' in session and session['current_task_id'] in processing_tasks:
            return jsonify(dict(processing_tasks[session['current_task_id']]))
        
        # Check if any tasks exist at all
        if processing_tasks:
            # Find most recent task as fallback
            most_recent_task = max(processing_tasks.items(), 
                                   key=lambda x: x[1].get('timestamp', 0) 
                                   if isinstance(x[1], dict) else 0)
            return jsonify({
                'status': 'recovered',
                'message': 'Task ID not found, showing most recent task',
                'task_id': most_recent_task[0],
                **most_recent_task[1]
            })
    
    # Default response if no tasks are found
    return jsonify({
        'status': 'unknown', 
        'message': 'No active tasks found. Please try uploading a new video.'
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static/results', filename, as_attachment=True)

# Cleanup old tasks periodically (keep for 24 hours)
@app.before_request
def cleanup_old_tasks():
    if random.random() < 0.01:  # 1% chance to run on any request
        with task_lock:
            current_time = time.time()
            task_ids = list(processing_tasks.keys())
            for task_id in task_ids:
                task = processing_tasks[task_id]
                if isinstance(task, dict) and 'timestamp' in task:
                    # Remove tasks older than 24 hours
                    if current_time - task['timestamp'] > 86400:
                        del processing_tasks[task_id]

if __name__ == '__main__':
    import random  # For random sampling in cleanup
    
    # Disable watchdog reloading for site-packages
    import os
    import sys
    
    # Set environment variable to disable auto-reloading for specific directories
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Run app with limited or no reloading
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    # Option 1: Run without debug mode (no auto-reload)
    # app.run(debug=False, host='0.0.0.0', port=5000)
    
    # Option 2: Run with debug but disable reloader
    # app.run(debug=True, use_reloader=False)
    
    # Option 3: Run with debug and selective reloading (preferred)
    from werkzeug.serving import run_simple
    extra_files = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    ]
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True,
               extra_files=extra_files, reloader_interval=1,
               static_files={'/static': 'static'})
