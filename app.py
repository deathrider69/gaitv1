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
import mediapipe as mp
from scipy import ndimage
from skimage import morphology, measure
import matplotlib.pyplot as plt

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

class GaitAnalyzer:
    """Class for extracting gait features from person videos"""
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_silhouette_and_gei(self, video_path, output_dir, person_name):
        """
        Extract gait silhouettes and create Gait Energy Image (GEI)
        
        Args:
            video_path: Path to individual person video
            output_dir: Directory to save outputs
            person_name: Name of the person for organizing outputs
        """
        # Create subdirectories for gait analysis
        gait_dir = os.path.join(output_dir, "gait_analysis", person_name)
        silhouettes_dir = os.path.join(gait_dir, "silhouettes")
        gei_dir = os.path.join(gait_dir, "gei")
        
        for dir_path in [gait_dir, silhouettes_dir, gei_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        silhouettes = []
        frame_count = 0
        
        # Background subtractor for silhouette extraction
        back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        print(f"Extracting silhouettes for {person_name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Method 1: Background subtraction for silhouette
            fg_mask = back_sub.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Method 2: Use MediaPipe for better segmentation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.segmentation_mask is not None:
                # Use MediaPipe segmentation mask
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
                silhouette_mask = np.where(condition, 255, 0).astype(np.uint8)[:, :, 0]
            else:
                # Fallback to background subtraction
                silhouette_mask = fg_mask
            
            # Resize silhouette to standard size (128x64 is common for gait analysis)
            silhouette_resized = cv2.resize(silhouette_mask, (64, 128))
            
            # Save individual silhouette
            silhouette_path = os.path.join(silhouettes_dir, f"silhouette_{frame_count:05d}.png")
            cv2.imwrite(silhouette_path, silhouette_resized)
            
            silhouettes.append(silhouette_resized)
            frame_count += 1
        
        cap.release()
        
        if len(silhouettes) > 0:
            # Create Gait Energy Image (GEI)
            gei = self.create_gei(silhouettes)
            
            # Save GEI
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gei_path = os.path.join(gei_dir, f"gei_{person_name}_{timestamp}.png")
            cv2.imwrite(gei_path, gei)
            
            # Create and save normalized GEI
            gei_normalized = cv2.normalize(gei, None, 0, 255, cv2.NORM_MINMAX)
            gei_norm_path = os.path.join(gei_dir, f"gei_normalized_{person_name}_{timestamp}.png")
            cv2.imwrite(gei_norm_path, gei_normalized)
            
            print(f"Created GEI for {person_name}: {gei_path}")
            
            return {
                'silhouettes_count': len(silhouettes),
                'silhouettes_dir': silhouettes_dir,
                'gei_path': gei_path,
                'gei_normalized_path': gei_norm_path
            }
        
        return None
    
    def create_gei(self, silhouettes):
        """
        Create Gait Energy Image from silhouettes
        GEI is the average of all silhouettes in a gait cycle
        """
        if len(silhouettes) == 0:
            return None
        
        # Convert to float for averaging
        silhouettes_array = np.array(silhouettes, dtype=np.float32)
        
        # Calculate average (GEI)
        gei = np.mean(silhouettes_array, axis=0)
        
        return gei.astype(np.uint8)
    
    def extract_skeletal_features(self, video_path, output_dir, person_name):
        """
        Extract skeletal gait features using MediaPipe Pose
        """
        # Create directory for skeletal features
        skeletal_dir = os.path.join(output_dir, "gait_analysis", person_name, "skeletal")
        os.makedirs(skeletal_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # Store pose landmarks for each frame
        pose_sequences = []
        annotated_frames = []
        frame_count = 0
        
        print(f"Extracting skeletal features for {person_name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])
                
                pose_sequences.append({
                    'frame': frame_count,
                    'landmarks': landmarks,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else frame_count
                })
            
            annotated_frames.append(annotated_frame)
            frame_count += 1
        
        cap.release()
        
        if len(pose_sequences) > 0:
            # Save pose data as JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pose_data_path = os.path.join(skeletal_dir, f"pose_data_{person_name}_{timestamp}.json")
            
            with open(pose_data_path, 'w') as f:
                json.dump(pose_sequences, f, indent=2)
            
            # Create skeletal gait video
            skeletal_video_path = os.path.join(skeletal_dir, f"skeletal_{person_name}_{timestamp}.mp4")
            self.create_skeletal_video(annotated_frames, skeletal_video_path, cap.get(cv2.CAP_PROP_FPS))
            
            # Extract specific gait features
            gait_features = self.extract_gait_parameters(pose_sequences)
            
            # Save gait features
            features_path = os.path.join(skeletal_dir, f"gait_features_{person_name}_{timestamp}.json")
            with open(features_path, 'w') as f:
                json.dump(gait_features, f, indent=2)
            
            print(f"Extracted skeletal features for {person_name}")
            
            return {
                'pose_sequences_count': len(pose_sequences),
                'pose_data_path': pose_data_path,
                'skeletal_video_path': skeletal_video_path,
                'gait_features_path': features_path,
                'gait_features': gait_features
            }
        
        return None
    
    def create_skeletal_video(self, annotated_frames, output_path, fps):
        """Create video with skeletal annotations"""
        if len(annotated_frames) == 0:
            return
        
        height, width = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in annotated_frames:
            out.write(frame)
        
        out.release()
    
    def extract_gait_parameters(self, pose_sequences):
        """
        Extract specific gait parameters from pose sequences
        """
        if len(pose_sequences) < 10:  # Need sufficient frames
            return {}
        
        # Key landmark indices (MediaPipe Pose)
        NOSE = 0
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32
        
        gait_features = {
            'stride_length': [],
            'step_width': [],
            'cadence': 0,
            'swing_time': {'left': [], 'right': []},
            'stance_time': {'left': [], 'right': []},
            'joint_angles': {
                'left_knee': [],
                'right_knee': [],
                'left_hip': [],
                'right_hip': []
            },
            'center_of_mass': [],
            'vertical_displacement': []
        }
        
        try:
            for seq in pose_sequences:
                landmarks = seq['landmarks']
                
                if len(landmarks) >= 33:  # MediaPipe has 33 landmarks
                    # Calculate center of mass (approximated by hip midpoint)
                    left_hip = landmarks[LEFT_HIP]
                    right_hip = landmarks[RIGHT_HIP]
                    com_x = (left_hip[0] + right_hip[0]) / 2
                    com_y = (left_hip[1] + right_hip[1]) / 2
                    gait_features['center_of_mass'].append([com_x, com_y])
                    
                    # Vertical displacement (using nose as reference)
                    nose_y = landmarks[NOSE][1]
                    gait_features['vertical_displacement'].append(nose_y)
                    
                    # Calculate joint angles
                    left_knee_angle = self.calculate_angle(
                        landmarks[LEFT_HIP][:2],
                        landmarks[LEFT_KNEE][:2],
                        landmarks[LEFT_ANKLE][:2]
                    )
                    right_knee_angle = self.calculate_angle(
                        landmarks[RIGHT_HIP][:2],
                        landmarks[RIGHT_KNEE][:2],
                        landmarks[RIGHT_ANKLE][:2]
                    )
                    
                    gait_features['joint_angles']['left_knee'].append(left_knee_angle)
                    gait_features['joint_angles']['right_knee'].append(right_knee_angle)
            
            # Calculate summary statistics
            if gait_features['center_of_mass']:
                com_array = np.array(gait_features['center_of_mass'])
                gait_features['stride_length_mean'] = np.std(com_array[:, 0]) * 2  # Approximation
                gait_features['step_width_mean'] = np.std(com_array[:, 1]) * 2   # Approximation
            
            if gait_features['vertical_displacement']:
                gait_features['vertical_displacement_range'] = (
                    np.max(gait_features['vertical_displacement']) - 
                    np.min(gait_features['vertical_displacement'])
                )
            
            # Estimate cadence (steps per minute)
            if len(pose_sequences) > 1:
                total_time = pose_sequences[-1]['timestamp'] - pose_sequences[0]['timestamp']
                if total_time > 0:
                    # Rough estimation based on vertical displacement peaks
                    if gait_features['vertical_displacement']:
                        peaks = self.find_gait_cycles(gait_features['vertical_displacement'])
                        if len(peaks) > 1:
                            gait_features['cadence'] = (len(peaks) * 60) / total_time
            
        except Exception as e:
            print(f"Error extracting gait parameters: {str(e)}")
        
        return gait_features
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            p1 = np.array(point1)
            p2 = np.array(point2)
            p3 = np.array(point3)
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return 0.0
    
    def find_gait_cycles(self, displacement_data):
        """Find gait cycles from vertical displacement data"""
        try:
            # Simple peak detection
            displacement_array = np.array(displacement_data)
            
            # Find local maxima (peaks)
            peaks = []
            for i in range(1, len(displacement_array) - 1):
                if (displacement_array[i] > displacement_array[i-1] and 
                    displacement_array[i] > displacement_array[i+1]):
                    peaks.append(i)
            
            return peaks
        except:
            return []

class PersonTracker:
    def __init__(self, model_name="yolo11m.pt"):
        """Initialize the person tracker with YOLO model"""
        print("Loading YOLO model...")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
        
        # Initialize gait analyzer
        self.gait_analyzer = GaitAnalyzer()
        
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
                    progress = 50 + (len([p for p in person_frames.keys() if p in person_labels]) * 15 / len(person_labels))
                    progress_callback(progress, f"Creating frames for {label}")
        
        # Store individual video paths for gait analysis
        individual_video_paths = {}
        
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
                individual_video_paths[label] = video_path
                print(f"Created video for {label} with {len(frames)} frames")
        
        # Perform gait analysis on individual videos
        print("Performing gait analysis...")
        gait_results = {}
        output_base_dir = os.path.dirname(individual_videos_dir)
        
        for person_name, video_path in individual_video_paths.items():
            if progress_callback:
                progress = 75 + (len(gait_results) * 15 / len(individual_video_paths))
                progress_callback(progress, f"Analyzing gait for {person_name}")
            
            try:
                # Extract silhouettes and GEI
                silhouette_results = self.gait_analyzer.extract_silhouette_and_gei(
                    video_path, output_base_dir, person_name
                )
                
                # Extract skeletal features
                skeletal_results = self.gait_analyzer.extract_skeletal_features(
                    video_path, output_base_dir, person_name
                )
                
                gait_results[person_name] = {
                    'silhouette_analysis': silhouette_results,
                    'skeletal_analysis': skeletal_results
                }
                
                print(f"Completed gait analysis for {person_name}")
                
            except Exception as e:
                print(f"Error in gait analysis for {person_name}: {str(e)}")
                gait_results[person_name] = {'error': str(e)}
        
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
            'gait_analysis': gait_results,
            'output_dirs': {
                'individual_frames': individual_frames_dir,
                'individual_videos': individual_videos_dir,
                'full_frames': full_frames_dir,
                'gait_analysis': os.path.join(output_base_dir, "gait_analysis")
            }
        }
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {detection_results['frame_count']}")
        print(f"People detected: {len(person_frames)}")
        print(f"People labeled: {len(person_labels)}")
        print(f"Individual videos created: {final_results['individual_videos']}")
        print(f"Gait analysis completed for: {len(gait_results)} people")
        
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
    'detection_results': None,
    'sample_frames': {}
}

tracker = PersonTracker()

def update_progress(progress, message):
    """Update processing progress"""
    processing_status['progress'] = progress
    processing_status['message'] = message

# HTML Templates
INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Person Tracker & Gait Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 5px; background-color: #fafafa; }
        .upload-area:hover { border-color: #007bff; background-color: #f0f8ff; }
        input[type="file"] { margin: 20px 0; }
        button { background-color: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        .progress-container { margin: 20px 0; }
        .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #007bff; transition: width 0.3s ease; }
        .status { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .status.processing { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .status.complete { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .status.error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Person Tracker & Gait Analyzer</h1>
        
        <div class="upload-area">
            <h3>Upload Video</h3>
            <p>Select a video file to analyze people and their gait patterns</p>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="videoFile" name="video" accept="video/*" required>
                <br>
                <button type="submit" id="uploadBtn">Upload and Process</button>
            </form>
        </div>
        
        <div id="progressSection" class="hidden">
            <div class="progress-container">
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                </div>
                <div id="progressText">0% - Initializing...</div>
            </div>
        </div>
        
        <div id="statusSection" class="hidden">
            <div id="statusMessage" class="status"></div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a video file');
                return;
            }
            
            const formData = new FormData();
            formData.append('video', file);
            
            // Show progress section
            document.getElementById('progressSection').classList.remove('hidden');
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('uploadBtn').textContent = 'Processing...';
            
            // Upload file
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Start monitoring progress
                    monitorProgress();
                } else {
                    showStatus('error', 'Upload failed: ' + data.error);
                    resetUploadButton();
                }
            })
            .catch(error => {
                showStatus('error', 'Upload error: ' + error);
                resetUploadButton();
            });
        });
        
        function monitorProgress() {
            const interval = setInterval(() => {
                fetch('/progress')
                .then(response => response.json())
                .then(data => {
                    updateProgress(data.progress, data.message);
                    
                    if (data.progress >= 50 && data.detection_complete) {
                        clearInterval(interval);
                        // Redirect to labeling page
                        window.location.href = '/label';
                    } else if (!data.active && data.progress < 50) {
                        clearInterval(interval);
                        showStatus('error', 'Processing failed');
                        resetUploadButton();
                    }
                })
                .catch(error => {
                    clearInterval(interval);
                    showStatus('error', 'Error monitoring progress');
                    resetUploadButton();
                });
            }, 1000);
        }
        
        function updateProgress(progress, message) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = Math.round(progress) + '% - ' + message;
        }
        
        function showStatus(type, message) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.className = 'status ' + type;
            statusDiv.textContent = message;
            document.getElementById('statusSection').classList.remove('hidden');
        }
        
        function resetUploadButton() {
            document.getElementById('uploadBtn').disabled = false;
            document.getElementById('uploadBtn').textContent = 'Upload and Process';
        }
    </script>
</body>
</html>
'''

LABEL_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Label Detected People</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .people-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .person-card { border: 2px solid #ddd; border-radius: 10px; padding: 15px; text-align: center; background-color: #fafafa; }
        .person-image { max-width: 100%; height: 200px; object-fit: cover; border-radius: 5px; margin-bottom: 10px; }
        .person-input { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 5px; font-size: 14px; }
        button { background-color: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        .button-container { text-align: center; margin-top: 30px; }
        .progress-container { margin: 20px 0; }
        .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #28a745; transition: width 0.3s ease; }
        .status { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .status.processing { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .status.complete { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .hidden { display: none; }
        .instruction { background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Label Detected People</h1>
        
        <div class="instruction">
            <strong>Instructions:</strong> Please provide names for the detected people below. 
            Enter a unique name for each person (e.g., "John", "Person_A", etc.). 
            You can leave some fields empty if you don't want to analyze those people.
        </div>
        
        <div id="peopleSection">
            <div class="people-grid" id="peopleGrid">
                <!-- People cards will be inserted here -->
            </div>
            
            <div class="button-container">
                <button onclick="submitLabels()" id="submitBtn">Create Videos & Analyze Gait</button>
                <button onclick="goBack()">Back to Upload</button>
            </div>
        </div>
        
        <div id="processingSection" class="hidden">
            <div class="progress-container">
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                </div>
                <div id="progressText">Processing...</div>
            </div>
            <div id="statusMessage" class="status processing">Creating individual videos and analyzing gait patterns...</div>
        </div>
        
        <div id="resultsSection" class="hidden">
            <h2>Processing Complete!</h2>
            <div id="resultsSummary"></div>
            <div class="button-container">
                <button onclick="downloadResults()">Download Results</button>
                <button onclick="goBack()">Process Another Video</button>
            </div>
        </div>
    </div>

    <script>
        // Load detected people on page load
        window.onload = function() {
            loadDetectedPeople();
        };
        
        function loadDetectedPeople() {
            fetch('/get_detected_people')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayPeople(data.people);
                } else {
                    alert('Error loading detected people: ' + data.error);
                    window.location.href = '/';
                }
            })
            .catch(error => {
                alert('Error: ' + error);
                window.location.href = '/';
            });
        }
        
        function displayPeople(people) {
            const grid = document.getElementById('peopleGrid');
            grid.innerHTML = '';
            
            people.forEach(person => {
                const card = document.createElement('div');
                card.className = 'person-card';
                card.innerHTML = `
                    <img src="/sample_frame/${person.id}" alt="Person ${person.id}" class="person-image">
                    <h4>Person ${person.id}</h4>
                    <input type="text" class="person-input" id="label_${person.id}" 
                           placeholder="Enter name (e.g., John, Person_A)" maxlength="50">
                `;
                grid.appendChild(card);
            });
        }
        
        function submitLabels() {
            const labels = {};
            const inputs = document.querySelectorAll('.person-input');
            
            inputs.forEach(input => {
                const personId = input.id.replace('label_', '');
                const label = input.value.trim();
                if (label) {
                    labels[personId] = label;
                }
            });
            
            if (Object.keys(labels).length === 0) {
                alert('Please provide at least one name for the detected people.');
                return;
            }
            
            // Show processing section
            document.getElementById('peopleSection').classList.add('hidden');
            document.getElementById('processingSection').classList.remove('hidden');
            
            // Submit labels
            fetch('/submit_labels', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({labels: labels})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    monitorFinalProcessing();
                } else {
                    alert('Error submitting labels: ' + data.error);
                    showPeopleSection();
                }
            })
            .catch(error => {
                alert('Error: ' + error);
                showPeopleSection();
            });
        }
        
        function monitorFinalProcessing() {
            const interval = setInterval(() => {
                fetch('/progress')
                .then(response => response.json())
                .then(data => {
                    updateProgress(data.progress, data.message);
                    
                    if (data.progress >= 100 || !data.active) {
                        clearInterval(interval);
                        if (data.progress >= 100) {
                            showResults(data.final_results);
                        } else {
                            alert('Processing failed');
                            showPeopleSection();
                        }
                    }
                })
                .catch(error => {
                    clearInterval(interval);
                    alert('Error monitoring progress');
                    showPeopleSection();
                });
            }, 2000);
        }
        
        function updateProgress(progress, message) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = Math.round(progress) + '% - ' + message;
        }
        
        function showResults(results) {
            document.getElementById('processingSection').classList.add('hidden');
            document.getElementById('resultsSection').classList.remove('hidden');
            
            const summary = document.getElementById('resultsSummary');
            summary.innerHTML = `
                <div class="status complete">
                    <h3>Processing Summary:</h3>
                    <ul>
                        <li><strong>Total Frames Processed:</strong> ${results.total_frames}</li>
                        <li><strong>People Detected:</strong> ${results.people_detected}</li>
                        <li><strong>People Labeled:</strong> ${results.people_labeled}</li>
                        <li><strong>Individual Videos Created:</strong> ${results.individual_videos}</li>
                        <li><strong>Gait Analysis Completed For:</strong> ${Object.keys(results.gait_analysis).length} people</li>
                    </ul>
                    <p><strong>Output files have been saved to the server's output directory.</strong></p>
                </div>
            `;
        }
        
        function showPeopleSection() {
            document.getElementById('processingSection').classList.add('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            document.getElementById('peopleSection').classList.remove('hidden');
        }
        
        function downloadResults() {
            window.location.href = '/download_results';
        }
        
        function goBack() {
            window.location.href = '/';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page"""
    return INDEX_HTML

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    global processing_status
    
    if processing_status['active']:
        return jsonify({'success': False, 'error': 'Another video is currently being processed'})
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        video_path = os.path.join('uploads', filename)
        file.save(video_path)
        
        # Start processing in background thread
        processing_thread = threading.Thread(
            target=process_video_background, 
            args=(video_path,)
        )
        processing_thread.start()
        
        return jsonify({'success': True, 'message': 'Upload successful, processing started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_video_background(video_path):
    """Background processing function"""
    global processing_status
    
    try:
        processing_status['active'] = True
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting video processing...'
        
        # Process video to detect people
        detection_results = tracker.process_video(video_path, ".", update_progress)
        
        # Store results for labeling phase
        processing_status['detection_results'] = detection_results
        processing_status['sample_frames'] = detection_results['person_sample_frames']
        processing_status['detection_complete'] = True
        processing_status['progress'] = 50
        processing_status['message'] = 'Detection complete - ready for labeling'
        
    except Exception as e:
        processing_status['active'] = False
        processing_status['progress'] = 0
        processing_status['message'] = f'Error: {str(e)}'
        print(f"Processing error: {e}")

@app.route('/progress')
def get_progress():
    """Get current processing progress"""
    global processing_status
    return jsonify({
        'active': processing_status['active'],
        'progress': processing_status['progress'],
        'message': processing_status['message'],
        'detection_complete': processing_status.get('detection_complete', False),
        'final_results': processing_status.get('final_results', None)
    })

@app.route('/label')
def label_page():
    """Labeling page for detected people"""
    global processing_status
    
    if not processing_status.get('detection_complete', False):
        return redirect(url_for('index'))
    
    return LABEL_HTML

@app.route('/get_detected_people')
def get_detected_people():
    """Get detected people with sample frames"""
    global processing_status
    
    try:
        if not processing_status.get('sample_frames'):
            return jsonify({'success': False, 'error': 'No detected people available'})
        
        people = []
        for person_id in processing_status['sample_frames'].keys():
            people.append({'id': str(person_id)})
        
        return jsonify({'success': True, 'people': people})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sample_frame/<person_id>')
def get_sample_frame(person_id):
    """Get sample frame for a detected person"""
    global processing_status
    
    try:
        person_id = int(person_id)
        if person_id in processing_status['sample_frames']:
            sample_frame = processing_status['sample_frames'][person_id]
            
            # Save frame to temporary file
            temp_path = f"temp_frame_{person_id}.jpg"
            cv2.imwrite(temp_path, sample_frame)
            
            return send_file(temp_path, mimetype='image/jpeg')
        else:
            return "Frame not found", 404
            
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/submit_labels', methods=['POST'])
def submit_labels():
    """Submit person labels and start final processing"""
    global processing_status
    
    try:
        data = request.get_json()
        labels = data.get('labels', {})
        
        if not labels:
            return jsonify({'success': False, 'error': 'No labels provided'})
        
        # Convert string keys back to integers
        person_labels = {int(k): v for k, v in labels.items()}
        
        # Start final processing in background
        final_thread = threading.Thread(
            target=final_processing_background,
            args=(person_labels,)
        )
        final_thread.start()
        
        return jsonify({'success': True, 'message': 'Labels submitted, starting final processing'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def final_processing_background(person_labels):
    """Background function for final processing with labels"""
    global processing_status
    
    try:
        processing_status['progress'] = 50
        processing_status['message'] = 'Creating labeled videos and analyzing gait...'
        
        # Create labeled outputs and perform gait analysis
        final_results = tracker.create_labeled_outputs(
            processing_status['detection_results'],
            person_labels,
            update_progress
        )
        
        processing_status['final_results'] = convert_to_serializable(final_results)
        processing_status['progress'] = 100
        processing_status['message'] = 'Processing complete!'
        processing_status['active'] = False
        
    except Exception as e:
        processing_status['active'] = False
        processing_status['progress'] = 0
        processing_status['message'] = f'Error in final processing: {str(e)}'
        print(f"Final processing error: {e}")

@app.route('/download_results')
def download_results():
    """Download results (placeholder - could create zip file)"""
    return jsonify({'message': 'Results are available in the output directory on the server'})

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Person Tracker & Gait Analyzer")
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("Starting Person Tracker & Gait Analyzer Web Application...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()