#!/usr/bin/env python3
"""
Person Tracking Application
Detects, tracks, and extracts individual people from videos
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
import json
import shutil

def convert_to_serializable(obj, depth=0):
    try:
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            # Avoid converting large arrays to lists
            return f"<ndarray shape={obj.shape}, dtype={obj.dtype}>"
        if isinstance(obj, dict):
            if depth > 3:
                return "<nested dict truncated>"
            return {k: convert_to_serializable(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            if depth > 3 or len(obj) > 1000:
                return f"<list of length {len(obj)} truncated>"
            return [convert_to_serializable(i, depth + 1) for i in obj]
        return obj
    except Exception as e:
        return f"<error: {str(e)}>"

class PersonTracker:
    def __init__(self, model_name="yolo11m.pt"):
        """Initialize the person tracker with YOLO model"""
        print("Loading YOLO model...")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
        
    def process_video(self, video_path, base_output_dir=".", progress_callback=None):
        """
        Process video to track people and extract individual frames/videos
        
        Args:
            video_path: Path to input video
            base_output_dir: Base directory containing uploads/output folders
            progress_callback: Function to call with progress updates
        """
        
        # Create main directories if they don't exist
        uploads_dir = os.path.join(base_output_dir, "uploads")
        output_dir = os.path.join(base_output_dir, "output")
        
        # Create output subdirectories
        full_frames_dir = os.path.join(output_dir, "full_frames")
        individual_frames_dir = os.path.join(output_dir, "individual_frames")
        individual_videos_dir = os.path.join(output_dir, "individual_videos")
        
        for dir_path in [uploads_dir, output_dir, full_frames_dir, individual_frames_dir, individual_videos_dir]:
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
        person_sample_frames = {}  # Store sample frames for each person
        
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
                        # Convert numpy int32 to regular Python int
                        track_id = int(track_id)
                        
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
                        
                        # Store for later processing
                        person_frames[track_id].append(person_img)
                        person_bbox_history[track_id].append((x1_padded, y1_padded, x2_padded, y2_padded))
                        
                        # Store sample frame (first good frame of each person)
                        if track_id not in person_sample_frames and person_img.shape[0] > 50 and person_img.shape[1] > 50:
                            person_sample_frames[track_id] = person_img.copy()
            
            # Save annotated frame with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(full_frames_dir, f"frame_{timestamp}_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, clean_frame)
            
            frame_count += 1
            
            # Progress callback
            if progress_callback:
                progress = (frame_count / total_frames) * 50  # First 50% for detection
                progress_callback(progress, f"Detecting people in frame {frame_count}/{total_frames}")
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Return data for labeling phase
        results = {
            'person_frames': person_frames,
            'person_sample_frames': person_sample_frames,
            'frame_count': frame_count,
            'fps': fps,
            'video_dimensions': (width, height),
            'individual_frames_dir': individual_frames_dir,
            'individual_videos_dir': individual_videos_dir,
            'full_frames_dir': full_frames_dir
        }
        
        return results
    
    def create_labeled_outputs(self, detection_results, person_labels, progress_callback=None):
        """
        Create labeled individual frames and videos after user labeling
        """
        person_frames = detection_results['person_frames']
        fps = detection_results['fps']
        individual_frames_dir = detection_results['individual_frames_dir']
        individual_videos_dir = detection_results['individual_videos_dir']
        full_frames_dir = detection_results['full_frames_dir']
        width, height = detection_results['video_dimensions']
        
        # Create individual frames with user labels
        print("Creating labeled individual frames...")
        for person_id, frames in person_frames.items():
            if person_id in person_labels:
                label = person_labels[person_id]
                
                # Check if label already exists and create versioned folder
                person_dir = self._get_unique_person_dir(individual_frames_dir, label)
                os.makedirs(person_dir, exist_ok=True)
                
                # Save individual frames
                for i, person_img in enumerate(frames):
                    person_path = os.path.join(person_dir, f"frame_{i:05d}.jpg")
                    cv2.imwrite(person_path, person_img)
                
                if progress_callback:
                    progress = 50 + (len([p for p in person_frames.keys() if p in person_labels]) * 25 / len(person_labels))
                    progress_callback(progress, f"Creating frames for {label}")
        
        # Create individual videos with user labels
        print("Creating labeled individual videos...")
        for person_id, frames in person_frames.items():
            if person_id in person_labels and len(frames) > 5:  # Only create videos with sufficient frames
                label = person_labels[person_id]
                
                # Find max dimensions for this person's frames
                max_height = max(frame.shape[0] for frame in frames)
                max_width = max(frame.shape[1] for frame in frames)
                
                # Check if label already exists and create versioned filename
                video_filename = self._get_unique_video_filename(individual_videos_dir, label)
                video_path = os.path.join(individual_videos_dir, video_filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (max_width, max_height))
                
                for frame in frames:
                    resized_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    h, w = frame.shape[:2]
                    resized_frame[0:h, 0:w] = frame
                    video_writer.write(resized_frame)
                
                video_writer.release()
                print(f"Created video for {label} with {len(frames)} frames")
        
        # Create full tracking video with timestamp
        print("Creating full tracking video...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_video_path = os.path.join(os.path.dirname(individual_videos_dir), f"full_tracking_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(full_video_path, fourcc, fps, (width, height))
        
        annotated_frames = sorted([f for f in os.listdir(full_frames_dir) if f.endswith('.jpg')])
        for frame_file in annotated_frames:
            frame_path = os.path.join(full_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        
        if progress_callback:
            progress_callback(100, "Processing complete!")
        
        final_results = {
            'total_frames': detection_results['frame_count'],
            'people_detected': len(person_frames),
            'people_labeled': len(person_labels),
            'individual_videos': len([p for p in person_frames.keys() if p in person_labels and len(person_frames[p]) > 5]),
            'full_video': full_video_path,
            'output_dirs': {
                'individual_frames': individual_frames_dir,
                'individual_videos': individual_videos_dir,
                'full_frames': full_frames_dir
            }
        }
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {detection_results['frame_count']}")
        print(f"People detected: {len(person_frames)}")
        print(f"People labeled: {len(person_labels)}")
        print(f"Individual videos created: {final_results['individual_videos']}")
        
        return final_results
    
    def _get_unique_person_dir(self, base_dir, label):
        """Get unique directory name for person, handling duplicates"""
        person_dir = os.path.join(base_dir, label)
        if not os.path.exists(person_dir):
            return person_dir
        
        # If directory exists, find next available number
        counter = 2
        while True:
            versioned_dir = os.path.join(base_dir, f"{label}_{counter}")
            if not os.path.exists(versioned_dir):
                return versioned_dir
            counter += 1
    
    def _get_unique_video_filename(self, base_dir, label):
        """Get unique video filename, handling duplicates"""
        video_filename = f"{label}.mp4"
        if not os.path.exists(os.path.join(base_dir, video_filename)):
            return video_filename
        
        # If file exists, find next available number
        counter = 2
        while True:
            versioned_filename = f"{label}_{counter}.mp4"
            if not os.path.exists(os.path.join(base_dir, versioned_filename)):
                return versioned_filename
            counter += 1

# Flask Web Application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variables for tracking processing status
processing_status = {
    'active': False, 
    'progress': 0, 
    'message': '', 
    'stage': 'idle',  # idle, detecting, labeling, creating_outputs
    'detection_results': None,
    'final_results': None
}
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
        <title>Person Tracking App</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { border-color: #007bff; }
            .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .progress-fill { height: 100%; background-color: #007bff; transition: width 0.3s; }
            .results { background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .person-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 8px; display: inline-block; text-align: center; }
            .person-image { max-width: 150px; max-height: 150px; border-radius: 5px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #ccc; cursor: not-allowed; }
            input[type="text"] { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            .labeling-section { display: none; }
            .person-grid { display: flex; flex-wrap: wrap; justify-content: center; }
        </style>
    </head>
    <body>
        <h1>Person Tracking App</h1>
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
        
        <div id="labelingSection" class="labeling-section">
            <h3>Label Detected People</h3>
            <p>Please enter names for each detected person:</p>
            <div id="personGrid" class="person-grid"></div>
            <button id="submitLabelsBtn" onclick="submitLabels()">Create Labeled Videos</button>
        </div>
        
        <div id="resultsSection" style="display: none;">
            <h3>Processing Complete!</h3>
            <div class="results" id="results"></div>
            <button onclick="location.reload()">Process Another Video</button>
        </div>
        
        <script>
            let detectedPeople = {};
            
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
                document.getElementById('labelingSection').style.display = 'none';
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
                        } else if (data.stage === 'labeling') {
                            showLabelingInterface(data.detected_people);
                        } else if (data.final_results) {
                            showResults(data.final_results);
                        }
                    });
            }
            
            function showLabelingInterface(people) {
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('labelingSection').style.display = 'block';
                
                detectedPeople = people;
                const grid = document.getElementById('personGrid');
                grid.innerHTML = '';
                
                people.forEach(personId => {
                    const card = document.createElement('div');
                    card.className = 'person-card';
                    card.innerHTML = `
                        <img src="/person_sample/${personId}" class="person-image" alt="Person ${personId}">
                        <br>
                        <label>Person ${personId}:</label>
                        <br>
                        <input type="text" id="label_${personId}" placeholder="Enter name" required>
                    `;
                    grid.appendChild(card);
                });
            }
            
            function submitLabels() {
                const labels = {};
                let allFilled = true;
                
                detectedPeople.forEach(personId => {
                    const input = document.getElementById(`label_${personId}`);
                    if (input.value.trim()) {
                        labels[personId] = input.value.trim();
                    } else {
                        allFilled = false;
                    }
                });
                
                if (!allFilled) {
                    alert('Please enter names for all detected people');
                    return;
                }
                
                document.getElementById('submitLabelsBtn').disabled = true;
                document.getElementById('progressSection').style.display = 'block';
                document.getElementById('labelingSection').style.display = 'none';
                
                fetch('/submit_labels', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({labels: labels})
                }).then(response => response.json())
                  .then(data => {
                      if (data.success) {
                          checkProgress();
                      } else {
                          alert('Label submission failed: ' + data.message);
                      }
                  });
            }
            
            function showResults(results) {
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('resultsSection').style.display = 'block';
                document.getElementById('submitBtn').disabled = false;
                
                const resultsHtml = `
                    <p><strong>Total Frames:</strong> ${results.total_frames}</p>
                    <p><strong>People Detected:</strong> ${results.people_detected}</p>
                    <p><strong>People Labeled:</strong> ${results.people_labeled}</p>
                    <p><strong>Individual Videos:</strong> ${results.individual_videos}</p>
                    <p><strong>Output Directories:</strong></p>
                    <ul>
                        <li>Individual Frames: ${results.output_dirs.individual_frames}</li>
                        <li>Individual Videos: ${results.output_dirs.individual_videos}</li>
                        <li>Full Frames: ${results.output_dirs.full_frames}</li>
                    </ul>
                    <p><strong>Full Tracking Video:</strong> ${results.full_video}</p>
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
    
    # Save uploaded file with timestamp
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{filename}"
    
    # Create uploads directory
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    filepath = os.path.join(uploads_dir, filename)
    file.save(filepath)
    
    # Start processing in background thread
    processing_status = {
        'active': True, 
        'progress': 0, 
        'message': 'Starting detection...', 
        'stage': 'detecting',
        'detection_results': None,
        'final_results': None
    }
    
    def process_video_thread():
        global processing_status, tracker
        try:
            if tracker is None:
                tracker = PersonTracker()
            
            # Run detection phase
            detection_results = tracker.process_video(filepath, ".", progress_callback)
            
            # Convert detected people keys to regular Python integers for JSON serialization
            detected_people = [int(person_id) for person_id in detection_results['person_sample_frames'].keys()]
            
            # Move to labeling stage
            processing_status = {
                'active': False,
                'progress': 50,
                'message': 'Detection complete. Please label people.',
                'stage': 'labeling',
                'detection_results': detection_results,
                'detected_people': detected_people,
                'final_results': None
            }
            
        except Exception as e:
            processing_status = {
                'active': False, 
                'progress': 0, 
                'message': f'Error: {str(e)}', 
                'stage': 'error',
                'detection_results': None,
                'final_results': None
            }
    
    thread = threading.Thread(target=process_video_thread)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Processing started'})

@app.route('/person_sample/<int:person_id>')
def get_person_sample(person_id):
    """Serve sample image for person labeling"""
    if (processing_status['stage'] == 'labeling' and 
        processing_status['detection_results'] and 
        person_id in processing_status['detection_results']['person_sample_frames']):
        
        sample_frame = processing_status['detection_results']['person_sample_frames'][person_id]
        
        # Convert to jpg and serve
        _, buffer = cv2.imencode('.jpg', sample_frame)
        
        from flask import Response
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    else:
        return "Sample not found", 404

@app.route('/submit_labels', methods=['POST'])
def submit_labels():
    """Handle label submission and create final outputs"""
    global processing_status, tracker
    
    if processing_status['stage'] != 'labeling':
        return jsonify({'success': False, 'message': 'Not in labeling stage'})
    
    data = request.get_json()
    person_labels = {int(k): v for k, v in data['labels'].items()}
    
    # Start final processing
    processing_status['active'] = True
    processing_status['stage'] = 'creating_outputs'
    processing_status['message'] = 'Creating labeled outputs...'
    
    def create_outputs_thread():
        global processing_status, tracker
        try:
            final_results = tracker.create_labeled_outputs(
                processing_status['detection_results'], 
                person_labels, 
                progress_callback
            )
            
            processing_status = {
                'active': False,
                'progress': 100,
                'message': 'Complete!',
                'stage': 'complete',
                'detection_results': None,
                'final_results': final_results
            }
            
        except Exception as e:
            processing_status = {
                'active': False,
                'progress': 0,
                'message': f'Error: {str(e)}',
                'stage': 'error',
                'detection_results': None,
                'final_results': None
            }
    
    thread = threading.Thread(target=create_outputs_thread)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Creating outputs...'})

@app.route('/progress')
def get_progress():
    """Get current processing progress"""
    safe_status = convert_to_serializable(processing_status)
    return jsonify(safe_status)


def run_cli(video_path):
    """Run in command line mode"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return
    
    tracker = PersonTracker()
    
    # Copy video to uploads with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{os.path.basename(video_path)}"
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    copied_path = os.path.join(uploads_dir, filename)
    shutil.copy2(video_path, copied_path)
    
    # Run detection
    detection_results = tracker.process_video(copied_path, ".")
    
    # Get labels from user
    person_labels = {}
    print("\nDetected people:")
    for person_id, sample_frame in detection_results['person_sample_frames'].items():
        # Save sample frame temporarily for user to see
        temp_path = f"temp_person_{person_id}.jpg"
        cv2.imwrite(temp_path, sample_frame)
        print(f"Person {person_id} sample saved as {temp_path}")
        
        while True:
            label = input(f"Enter name for Person {person_id}: ").strip()
            if label:
                person_labels[person_id] = label
                break
            print("Please enter a valid name.")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Create final outputs
    final_results = tracker.create_labeled_outputs(detection_results, person_labels)
    
    print(f"\nResults:")
    print(f"Individual Frames: {final_results['output_dirs']['individual_frames']}")
    print(f"Individual Videos: {final_results['output_dirs']['individual_videos']}")
    print(f"Full Tracking Video: {final_results['full_video']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Person Tracking Application')
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