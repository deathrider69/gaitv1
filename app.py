#!/usr/bin/env python3
"""
Person Tracking
To detect, track, and extract individual people from videos
"""

import os
import sys
import argparse
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from flask import Flask, render_template_string, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import threading
import time

class PersonTracker:
    def __init__(self, model_name="yolo11m.pt"):
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
        
    def process_video(self, video_path, output_base_dir="output", progress_callback=None):
        
        #Process video to track people and extract individual frames/videos
        
        """
        Args:
            video_path: Path to input video
            output_base_dir: Base directory for outputs
            progress_callback: Function to call with progress updates
        """
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = os.path.join(output_base_dir, f"person_tracking_output_{timestamp}")
        
        # Create output directories
        os.makedirs(main_output_dir, exist_ok=True)
        frames_dir = os.path.join(main_output_dir, "full_frames")
        individual_frames_dir = os.path.join(main_output_dir, "individual_frames")
        individual_videos_dir = os.path.join(main_output_dir, "individual_videos")
        
        for dir_path in [frames_dir, individual_frames_dir, individual_videos_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize tracking variables
        frame_count = 0
        person_frames = defaultdict(list)
        person_bbox_history = defaultdict(list)
        
        # Process video frame by frame
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Run tracking on frame (only detect people - class 0)
            results = self.model.track(frame, persist=True, classes=[0])
            
            # Create clean frame copy
            clean_frame = frame.copy()
            
            # Process detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                cls = results[0].boxes.cls.cpu().numpy()
                
                for box, track_id, class_id in zip(boxes, track_ids, cls):
                    if class_id == 0:  # Person class
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Draw bounding box
                        cv2.rectangle(clean_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(clean_frame, f"Person {track_id}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add padding to bounding box
                        padding = 10
                        x1_padded = max(0, x1 - padding)
                        y1_padded = max(0, y1 - padding)
                        x2_padded = min(width, x2 + padding)
                        y2_padded = min(height, y2 + padding)
                        
                        # Extract person from frame
                        person_img = frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()
                        
                        # Create person directory
                        person_dir = os.path.join(individual_frames_dir, f"person_{track_id}")
                        os.makedirs(person_dir, exist_ok=True)
                        
                        # Save individual frame
                        person_path = os.path.join(person_dir, f"frame_{frame_count:05d}.jpg")
                        cv2.imwrite(person_path, person_img)
                        
                        # Store for video creation
                        person_frames[track_id].append(person_img)
                        person_bbox_history[track_id].append((x1_padded, y1_padded, x2_padded, y2_padded))
            
            # Save annotated frame
            output_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, clean_frame)
            
            frame_count += 1
            
            # Progress callback
            if progress_callback:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Create individual videos
        print("Creating individual videos...")
        for person_id, frames in person_frames.items():
            if len(frames) > 5:  # Only create videos with sufficient frames
                max_height = max(frame.shape[0] for frame in frames)
                max_width = max(frame.shape[1] for frame in frames)
                
                video_path = os.path.join(individual_videos_dir, f"person_{person_id}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (max_width, max_height))
                
                for frame in frames:
                    resized_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    h, w = frame.shape[:2]
                    resized_frame[0:h, 0:w] = frame
                    video_writer.write(resized_frame)
                
                video_writer.release()
                print(f"Created video for person {person_id} with {len(frames)} frames")
        
        # Create full tracking video
        print("Creating full tracking video...")
        full_video_path = os.path.join(main_output_dir, "full_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(full_video_path, fourcc, fps, (width, height))
        
        annotated_frames = sorted(os.listdir(frames_dir))
        for frame_file in annotated_frames:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        
        results = {
            'output_dir': main_output_dir,
            'total_frames': frame_count,
            'people_tracked': len(person_frames),
            'individual_videos': len([p for p in person_frames.keys() if len(person_frames[p]) > 5]),
            'full_video': full_video_path
        }
        
        print(f"\nProcessing complete!")
        print(f"Output directory: {main_output_dir}")
        print(f"Total frames processed: {frame_count}")
        print(f"People tracked: {len(person_frames)}")
        print(f"Individual videos created: {results['individual_videos']}")
        
        return results

# Flask Web Application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variables for tracking processing status
processing_status = {'active': False, 'progress': 0, 'message': '', 'results': None}
tracker = None

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def progress_callback(progress, message):
    """Update processing progress"""
    processing_status['progress'] = progress
    processing_status['message'] = message

@app.route('/')
def index():
    """Main page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Person Tracking</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { border-color: #007bff; }
            .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .progress-fill { height: 100%; background-color: #007bff; transition: width 0.3s; }
            .results { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #ccc; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <h1>Person Tracking</h1>
        <p>Upload a video to detect, track, and extract individual people.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" id="videoFile" name="video" accept="video/*" required>
                <p>Select a video file (MP4, AVI, MOV, etc.)</p>
            </div>
            <button type="submit" id="submitBtn">Process Video</button>
        </form>
        
        <div id="progressSection" style="display: none;">
            <h3>Processing...</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p id="progressMessage">Starting...</p>
        </div>
        
        <div id="resultsSection" style="display: none;">
            <h3>Processing Complete!</h3>
            <div class="results" id="results"></div>
            <button onclick="location.reload()">Process Another Video</button>
        </div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('videoFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a video file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('video', file);
                
                document.getElementById('submitBtn').disabled = true;
                document.getElementById('progressSection').style.display = 'block';
                document.getElementById('resultsSection').style.display = 'none';
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                }).then(response => response.json())
                  .then(data => {
                      if (data.success) {
                          checkProgress();
                      } else {
                          alert('Upload failed: ' + data.message);
                          document.getElementById('submitBtn').disabled = false;
                          document.getElementById('progressSection').style.display = 'none';
                      }
                  });
            };
            
            function checkProgress() {
                fetch('/progress')
                    .then(response => response.json())
                    .then(data => {
                        if (data.active) {
                            document.getElementById('progressFill').style.width = data.progress + '%';
                            document.getElementById('progressMessage').textContent = data.message;
                            setTimeout(checkProgress, 1000);
                        } else if (data.results) {
                            showResults(data.results);
                        }
                    });
            }
            
            function showResults(results) {
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('resultsSection').style.display = 'block';
                document.getElementById('submitBtn').disabled = false;
                
                const resultsHtml = `
                    <p><strong>Total Frames:</strong> ${results.total_frames}</p>
                    <p><strong>People Tracked:</strong> ${results.people_tracked}</p>
                    <p><strong>Individual Videos:</strong> ${results.individual_videos}</p>
                    <p><strong>Output Directory:</strong> ${results.output_dir}</p>
                    <p><a href="/download/${encodeURIComponent(results.output_dir)}" target="_blank">Download Results</a></p>
                `;
                document.getElementById('results').innerHTML = resultsHtml;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    global processing_status, tracker
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type'})
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    # Start processing in background thread
    processing_status = {'active': True, 'progress': 0, 'message': 'Starting...', 'results': None}
    
    def process_video_thread():
        global processing_status, tracker
        try:
            if tracker is None:
                tracker = PersonTracker()
            
            results = tracker.process_video(filepath, app.config['OUTPUT_FOLDER'], progress_callback)
            processing_status = {'active': False, 'progress': 100, 'message': 'Complete', 'results': results}
        except Exception as e:
            processing_status = {'active': False, 'progress': 0, 'message': f'Error: {str(e)}', 'results': None}
    
    thread = threading.Thread(target=process_video_thread)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Processing started'})

@app.route('/progress')
def get_progress():
    """Get current processing progress"""
    return jsonify(processing_status)

@app.route('/download/<path:output_dir>')
def download_results(output_dir):
    """Serve results directory for download (simplified)"""
    # In a production app, you might want to create a zip file
    # For now, just return the full tracking video
    full_video_path = os.path.join(output_dir, "full_tracking.mp4")
    if os.path.exists(full_video_path):
        return send_file(full_video_path, as_attachment=True)
    else:
        return "Results not found", 404

def run_cli(video_path):
    """Run in command line mode"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return
    
    tracker = PersonTracker()
    results = tracker.process_video(video_path)
    
    print(f"\nResults saved to: {results['output_dir']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Person Tracking')
    parser.add_argument('video', nargs='?', help='Path to video file (CLI mode)')
    parser.add_argument('--web', action='store_true', help='Run web interface')
    parser.add_argument('--port', type=int, default=5000, help='Port for web interface')
    
    args = parser.parse_args()
    
    if args.web or (not args.video and len(sys.argv) == 1):
        # Web interface mode
        print("Starting web interface...")
        print(f"Open your browser to: http://localhost:{args.port}")
        app.run(debug=True, port=args.port, host='0.0.0.0')
    elif args.video:
        # CLI mode
        run_cli(args.video)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()