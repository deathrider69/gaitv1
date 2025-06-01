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
    """Streamlined class for real-time gait analysis from person videos"""
    import cv2
    import mediapipe as mp
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import os
    from datetime import datetime
    def __init__(self):
        # Initialize MediaPipe Pose with world landmarks enabled
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Required for world landmarks
            enable_segmentation=False,  # Disabled - not needed for gait analysis
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def analyze_gait(self, video_path, output_dir, person_name, display_video=False, save_annotated_video=True):
        """
        Analyze gait patterns from person video
        
        Args:
            video_path: Path to individual person video
            output_dir: Directory to save outputs
            person_name: Name of the person for organizing outputs
            display_video: Whether to display video during processing
            save_annotated_video: Whether to save annotated video output
        """
        # Create output directories
        analysis_dir = os.path.join(output_dir, "gait_analysis", person_name)
        os.makedirs(analysis_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dt = 1.0 / fps if fps > 0 else 1.0/30.0
        
        print(f"Analyzing gait for {person_name}...")
        print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Initialize storage for gait metrics
        gait_data = {
            'left_knee_angles': [],
            'right_knee_angles': [],
            'left_hip_angles': [],
            'right_hip_angles': [],
            'shoulder_widths': [],
            'torso_lengths': [],
            'hip_widths': [],
            'heights': [],
            'frame_timestamps': [],
            'center_of_mass': [],
            'vertical_displacement': []
        }
        
        # Setup video writer for annotated output
        annotated_video_path = None
        out = None
        if save_annotated_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_video_path = os.path.join(analysis_dir, f"annotated_{person_name}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        time_elapsed = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                
                if not results.pose_landmarks:
                    frame_count += 1
                    time_elapsed += dt
                    continue
                
                # Extract landmarks
                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                
                # Helper function for 2D coordinates
                def xy(idx):
                    return (int(lm[idx].x * w), int(lm[idx].y * h))
                
                # Key landmarks
                L_hip = xy(23); L_knee = xy(25); L_ankle = xy(27)
                R_hip = xy(24); R_knee = xy(26); R_ankle = xy(28)
                L_shoulder = xy(11); R_shoulder = xy(12)
                nose = xy(0)
                
                # Calculate joint angles
                left_knee_angle = self.calculate_angle(L_hip, L_knee, L_ankle)
                right_knee_angle = self.calculate_angle(R_hip, R_knee, R_ankle)
                
                # Calculate hip angles (hip-knee-ankle for thigh angle)
                left_hip_angle = self.calculate_angle(L_shoulder, L_hip, L_knee)
                right_hip_angle = self.calculate_angle(R_shoulder, R_hip, R_knee)
                
                # Calculate body measurements
                mid_shoulder = ((L_shoulder[0] + R_shoulder[0])/2, (L_shoulder[1] + R_shoulder[1])/2)
                mid_hip = ((L_hip[0] + R_hip[0])/2, (L_hip[1] + R_hip[1])/2)
                
                torso_length = np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip))
                hip_width = np.linalg.norm(np.array(L_hip) - np.array(R_hip))
                shoulder_width = np.linalg.norm(np.array(L_shoulder) - np.array(R_shoulder))
                normalized_shoulder_width = shoulder_width / torso_length if torso_length > 0 else 0
                
                # Height estimation using world landmarks
                total_height = 0
                if results.pose_world_landmarks:
                    world_lm = results.pose_world_landmarks.landmark
                    
                    # Get key points in 3D world coordinates (meters)
                    head = world_lm[self.mp_pose.PoseLandmark.NOSE]
                    heel_left = world_lm[self.mp_pose.PoseLandmark.LEFT_HEEL]
                    heel_right = world_lm[self.mp_pose.PoseLandmark.RIGHT_HEEL]
                    
                    # Calculate height components
                    head_height = abs(head.y)
                    leg_length = (abs(heel_left.y) + abs(heel_right.y)) / 2
                    total_height = head_height + leg_length
                
                # Store measurements
                gait_data['left_knee_angles'].append(left_knee_angle)
                gait_data['right_knee_angles'].append(right_knee_angle)
                gait_data['left_hip_angles'].append(left_hip_angle)
                gait_data['right_hip_angles'].append(right_hip_angle)
                gait_data['shoulder_widths'].append(normalized_shoulder_width)
                gait_data['torso_lengths'].append(torso_length)
                gait_data['hip_widths'].append(hip_width)
                gait_data['heights'].append(total_height)
                gait_data['frame_timestamps'].append(time_elapsed)
                gait_data['center_of_mass'].append([mid_hip[0]/w, mid_hip[1]/h])  # Normalized
                gait_data['vertical_displacement'].append(nose[1]/h)  # Normalized
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                # Add text annotations
                cv2.putText(frame, f"Norm Shoulder: {normalized_shoulder_width:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Knee: {left_knee_angle:.1f}°",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Knee: {right_knee_angle:.1f}°",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(frame, f"Est Height: {total_height:.2f}m",
                #           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save annotated frame
                if out is not None:
                    out.write(frame)
                
                # Display frame if requested
                if display_video:
                    cv2.imshow(f'Gait Analysis - {person_name}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                time_elapsed += dt
                
                # Progress indicator
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except Exception as e:
            print(f"Error during gait analysis: {str(e)}")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            if display_video:
                cv2.destroyAllWindows()
        
        # Calculate derived metrics
        analysis_results = self.calculate_gait_features(gait_data, person_name, dt)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw gait data
        data_path = os.path.join(analysis_dir, f"gait_data_{person_name}_{timestamp}.json")
        try:
            with open(data_path, 'w') as f:
                json.dump(gait_data, f, indent=2)
        except Exception as e:
            print(f"Error saving gait data: {e}")
        
        # Save analysis results
        results_path = os.path.join(analysis_dir, f"gait_analysis_{person_name}_{timestamp}.json")
        try:
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
        except Exception as e:
            print(f"Error saving analysis results: {e}")
        
        # Generate and save plots
        plots_path = os.path.join(analysis_dir, f"gait_plots_{person_name}_{timestamp}.png")
        self.create_gait_plots(gait_data, plots_path, person_name)
        
        print(f"Gait analysis complete for {person_name}")
        print(f"Processed {frame_count} frames")
        
        return {
            'person_name': person_name,
            'frames_processed': frame_count,
            'data_path': data_path,
            'results_path': results_path,
            'plots_path': plots_path,
            'annotated_video_path': annotated_video_path,
            'analysis_results': analysis_results
        }
    
    def calculate_gait_features(self, gait_data, person_name, dt):
        """Calculate advanced gait features from collected data"""
        
        if len(gait_data['left_knee_angles']) < 10:
            return {'error': 'Insufficient data for analysis'}
        
        analysis_results = {
            'person_name': person_name,
            'summary_statistics': {},
            'temporal_parameters': {},
            'kinematic_parameters': {},
            'stability_parameters': {}
        }
        
        try:
            # Calculate velocities and accelerations
            left_knee_vel = np.diff(gait_data['left_knee_angles']) / dt
            right_knee_vel = np.diff(gait_data['right_knee_angles']) / dt
            left_knee_acc = np.diff(left_knee_vel) / dt
            right_knee_acc = np.diff(right_knee_vel) / dt
            
            # Summary statistics
            features = {
                'left_knee_angle': gait_data['left_knee_angles'],
                'right_knee_angle': gait_data['right_knee_angles'],
                'left_hip_angle': gait_data['left_hip_angles'],
                'right_hip_angle': gait_data['right_hip_angles'],
                'normalized_shoulder_width': gait_data['shoulder_widths'],
                'torso_length': gait_data['torso_lengths'],
                'hip_width': gait_data['hip_widths'],
                'height': gait_data['heights'],
                'left_knee_velocity': left_knee_vel.tolist(),
                'right_knee_velocity': right_knee_vel.tolist(),
                'left_knee_acceleration': left_knee_acc.tolist(),
                'right_knee_acceleration': right_knee_acc.tolist()
            }
            
            for feature_name, values in features.items():
                if len(values) > 0:
                    analysis_results['summary_statistics'][feature_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'range': float(np.max(values) - np.min(values))
                    }
            
            # Temporal parameters
            if len(gait_data['frame_timestamps']) > 1:
                total_time = gait_data['frame_timestamps'][-1] - gait_data['frame_timestamps'][0]
                
                # Estimate gait cycles from vertical displacement
                vertical_disp = np.array(gait_data['vertical_displacement'])
                peaks = self.find_peaks(vertical_disp)
                
                if len(peaks) > 1:
                    cycle_times = np.diff(np.array(gait_data['frame_timestamps'])[peaks])
                    analysis_results['temporal_parameters'] = {
                        'estimated_gait_cycles': len(peaks),
                        'average_cycle_time': float(np.mean(cycle_times)) if len(cycle_times) > 0 else 0,
                        'cadence_steps_per_minute': (len(peaks) * 60) / total_time if total_time > 0 else 0,
                        'stride_frequency': len(peaks) / total_time if total_time > 0 else 0
                    }
            
            # Kinematic parameters
            if gait_data['heights']:
                height_stats = analysis_results['summary_statistics'].get('height', {})
                analysis_results['kinematic_parameters'] = {
                    'mean_estimated_height_m': height_stats.get('mean', 0),
                    'height_variability': height_stats.get('std', 0),
                    'knee_angle_asymmetry': abs(
                        analysis_results['summary_statistics']['left_knee_angle']['mean'] - 
                        analysis_results['summary_statistics']['right_knee_angle']['mean']
                    ),
                    'hip_angle_asymmetry': abs(
                        analysis_results['summary_statistics']['left_hip_angle']['mean'] - 
                        analysis_results['summary_statistics']['right_hip_angle']['mean']
                    ) if 'left_hip_angle' in analysis_results['summary_statistics'] else 0
                }
            
            # Stability parameters
            com_data = np.array(gait_data['center_of_mass'])
            if len(com_data) > 0:
                com_x_std = np.std(com_data[:, 0])
                com_y_std = np.std(com_data[:, 1])
                
                analysis_results['stability_parameters'] = {
                    'center_of_mass_variability_x': float(com_x_std),
                    'center_of_mass_variability_y': float(com_y_std),
                    'postural_stability_index': float(np.sqrt(com_x_std**2 + com_y_std**2)),
                    'vertical_displacement_range': float(np.ptp(gait_data['vertical_displacement']))
                }
        
        except Exception as e:
            analysis_results['error'] = f"Error in feature calculation: {str(e)}"
        
        return analysis_results
    
    def find_peaks(self, data, min_distance=10):
        """Simple peak detection"""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] > data[i-j] for j in range(1, min_distance+1)) and \
               all(data[i] > data[i+j] for j in range(1, min_distance+1)):
                peaks.append(i)
        return peaks
    
    def create_gait_plots(self, gait_data, output_path, person_name):
        """Create comprehensive gait analysis plots"""
        
        plt.figure(figsize=(15, 16))
        timestamps = gait_data['frame_timestamps']
        
        # Plot 1: Knee Angles
        plt.subplot(5, 1, 1)
        plt.plot(timestamps, gait_data['left_knee_angles'], 'b-', label='Left Knee', linewidth=2)
        plt.plot(timestamps, gait_data['right_knee_angles'], 'r-', label='Right Knee', linewidth=2)
        plt.title(f'Knee Angles During Gait - {person_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Hip Angles
        plt.subplot(5, 1, 2)
        if gait_data['left_hip_angles'] and gait_data['right_hip_angles']:
            plt.plot(timestamps, gait_data['left_hip_angles'], 'b-', label='Left Hip', linewidth=2)
            plt.plot(timestamps, gait_data['right_hip_angles'], 'r-', label='Right Hip', linewidth=2)
        plt.title('Hip Angles During Gait', fontsize=12, fontweight='bold')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Body Measurements
        plt.subplot(5, 1, 3)
        plt.plot(timestamps, gait_data['shoulder_widths'], 'g-', label='Normalized Shoulder Width', linewidth=2)
        plt.title('Normalized Shoulder Width', fontsize=12, fontweight='bold')
        plt.ylabel('Width/Torso Ratio')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Angular Velocities
        if len(timestamps) > 1:
            plt.subplot(5, 1, 4)
            dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1
            left_vel = np.diff(gait_data['left_knee_angles']) / dt
            right_vel = np.diff(gait_data['right_knee_angles']) / dt
            vel_timestamps = timestamps[:-1]
            
            plt.plot(vel_timestamps, left_vel, 'b-', label='Left Knee', linewidth=2)
            plt.plot(vel_timestamps, right_vel, 'r-', label='Right Knee', linewidth=2)
            plt.title('Knee Angular Velocities', fontsize=12, fontweight='bold')
            plt.ylabel('deg/s')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Height Estimation
        plt.subplot(5, 1, 5)
        if gait_data['heights']:
            plt.plot(timestamps[:len(gait_data['heights'])], gait_data['heights'], 'm-', linewidth=2)
            plt.title('Height Estimation Over Time', fontsize=12, fontweight='bold')
            plt.ylabel('Height (meters)')
        plt.xlabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gait analysis plots saved to: {output_path}")
    
    def print_analysis_summary(self, analysis_results):
        """Print a formatted summary of the gait analysis"""
        
        if 'error' in analysis_results:
            print(f"Analysis Error: {analysis_results['error']}")
            return
        
        print(f"\n{'='*50}")
        print(f"GAIT ANALYSIS SUMMARY - {analysis_results['person_name'].upper()}")
        print(f"{'='*50}")
        
        # Summary Statistics
        if 'summary_statistics' in analysis_results:
            print(f"\n{'='*20} FEATURE SUMMARY {'='*20}")
            stats = analysis_results['summary_statistics']
            
            for feature, values in stats.items():
                print(f"{feature:25s} mean = {values['mean']:6.2f}, std = {values['std']:6.2f}")
        
        # Temporal Parameters
        if 'temporal_parameters' in analysis_results:
            print(f"\n{'='*20} TEMPORAL PARAMETERS {'='*15}")
            temp_params = analysis_results['temporal_parameters']
            
            for param, value in temp_params.items():
                print(f"{param:30s}: {value:.2f}")
        
        # Kinematic Parameters
        if 'kinematic_parameters' in analysis_results:
            print(f"\n{'='*20} KINEMATIC PARAMETERS {'='*14}")
            kine_params = analysis_results['kinematic_parameters']
            
            for param, value in kine_params.items():
                print(f"{param:30s}: {value:.2f}")
        
        # Stability Parameters
        if 'stability_parameters' in analysis_results:
            print(f"\n{'='*20} STABILITY PARAMETERS {'='*15}")
            stab_params = analysis_results['stability_parameters']
            
            for param, value in stab_params.items():
                print(f"{param:30s}: {value:.4f}")
        
        print(f"\n{'='*50}")

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
                #silhouette_results = self.gait_analyzer.extract_silhouette_and_gei(
                #    video_path, output_base_dir, person_name
                #)
                
                # Extract skeletal features
                skeletal_results = self.gait_analyzer.analyze_gait(
                    video_path, output_base_dir, person_name
                )
                
                gait_results[person_name] = {
                    #'silhouette_analysis': silhouette_results,
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
            <strong>Instructions:</strong> Enter names for the below detected people. 
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
                           placeholder="Enter name" maxlength="50">
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