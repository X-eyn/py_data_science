import cv2
import numpy as np
import dlib
from typing import Tuple, Optional

class PupilDetector:
    def __init__(self, 
                 face_cascade_path: str = None, 
                 landmark_model_path: str = None):
        """
        Initialize pupil detection system with configurable models.
        
        Args:
            face_cascade_path: Path to Haar cascade classifier
            landmark_model_path: Path to dlib landmark predictor model
        """
        # Use default paths if not provided
        if face_cascade_path is None:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if landmark_model_path is None:
            landmark_model_path = "shape_predictor_68_face_landmarks.dat"
        
        # Load models
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.landmark_predictor = dlib.shape_predictor(landmark_model_path)
        
        # Adaptive detection parameters
        self.detection_params = {
            'pupil_min_ratio': 0.01,  # Minimum pupil size relative to eye region
            'pupil_max_ratio': 0.1,   # Maximum pupil size relative to eye region
            'staring_threshold': 0.25, # Threshold for determining staring
            'smoothing_factor': 0.2,  # Smoothing factor for pupil position (0.0 - 1.0)
        }
        
        # Initialize previous pupil positions for smoothing
        self.prev_left_pupil = None
        self.prev_right_pupil = None
        
        # Initialize state for staring detection
        self.staring_state = {
            "left": False,
            "right": False
        }

    def preprocess_eye_region(self, eye_roi_gray: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing of eye region for better pupil detection.
        
        Args:
            eye_roi_gray: Grayscale eye region image
        
        Returns:
            Preprocessed image optimized for pupil detection
        """
        # Histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(eye_roi_gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return blurred

    def detect_pupil(self, eye_roi_gray: np.ndarray, eye_landmarks: np.ndarray) -> Optional[Tuple]:
        """
        Advanced pupil detection with multiple detection strategies.
        
        Args:
            eye_roi_gray: Grayscale eye region image
            eye_landmarks: Landmarks defining the eye region
        
        Returns:
            Tuple of (pupil_center, pupil_radius) or None if not detected
        """
        # Preprocess eye region
        processed = self.preprocess_eye_region(eye_roi_gray)
        
        # Calculate dynamic area thresholds
        eye_width = np.max(eye_landmarks[:, 0]) - np.min(eye_landmarks[:, 0])
        eye_height = np.max(eye_landmarks[:, 1]) - np.min(eye_landmarks[:, 1])
        eye_area = eye_width * eye_height
        
        min_pupil_area = int(eye_area * self.detection_params['pupil_min_ratio'])
        max_pupil_area = int(eye_area * self.detection_params['pupil_max_ratio'])
        
        # Multiple thresholding methods
        threshold_methods = [
            # Otsu's thresholding
            cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            
            # Adaptive thresholding
            cv2.adaptiveThreshold(processed, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2)
        ]
        
        best_pupil_candidate = None
        best_score = float('inf')
        
        for thresh in threshold_methods:
            # Morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Filter contours by area
                if min_pupil_area < area < max_pupil_area:
                    # Fit ellipse requires at least 5 points
                    if len(cnt) >= 5:
                        try:
                            # Fit ellipse to contour
                            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                            
                            # Calculate circularity
                            perimeter = cv2.arcLength(cnt, True)
                            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            
                            # Scoring mechanism
                            score = abs(1 - circularity)  # Lower score means more circular
                            
                            if score < best_score:
                                best_score = score
                                best_pupil_candidate = (x, y, min(MA, ma) / 2)
                        except cv2.error:
                            # Skip if ellipse fitting fails
                            continue
        
        return best_pupil_candidate

    def detect_gaze(self, frame: np.ndarray) -> np.ndarray:
        """
        Main method to detect faces, eyes, and pupils, and determine gaze direction.
        
        Args:
            frame: Input video frame
        
        Returns:
            Frame with annotations
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            # Convert face region to dlib rectangle
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Detect facial landmarks
            landmarks = self.landmark_predictor(gray, dlib_rect)
            
            # Extract eye landmarks
            left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                           for i in range(36, 42)])
            right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                            for i in range(42, 48)])
            
            # Process each eye
            for i, eye_landmarks in enumerate([left_eye_landmarks, right_eye_landmarks]):
                # Calculate eye region
                x_min, y_min = np.min(eye_landmarks, axis=0)
                x_max, y_max = np.max(eye_landmarks, axis=0)
                
                # Add padding
                padding = 0.5
                eye_width = x_max - x_min
                eye_height = y_max - y_min
                
                x_min = max(0, int(x_min - padding * eye_width))
                y_min = max(0, int(y_min - padding * eye_height))
                x_max = min(frame.shape[1], int(x_max + padding * eye_width))
                y_max = min(frame.shape[0], int(y_max + padding * eye_height))
                
                # Extract eye regions
                eye_roi_gray = gray[y_min:y_max, x_min:x_max]
                eye_roi_color = frame[y_min:y_max, x_min:x_max]
                
                # Detect pupil
                pupil = self.detect_pupil(eye_roi_gray, eye_landmarks - [x_min, y_min])
                
                if pupil:
                    # Smooth the pupil position
                    x, y, radius = pupil

                    # Determine which eye we are processing and update the corresponding previous pupil position
                    if i == 0:  # Left eye
                        if self.prev_left_pupil is not None:
                            prev_x, prev_y, _ = self.prev_left_pupil
                            x = int((1 - self.detection_params['smoothing_factor']) * prev_x + self.detection_params['smoothing_factor'] * x)
                            y = int((1 - self.detection_params['smoothing_factor']) * prev_y + self.detection_params['smoothing_factor'] * y)
                        self.prev_left_pupil = (x, y, radius)
                    else:  # Right eye
                        if self.prev_right_pupil is not None:
                            prev_x, prev_y, _ = self.prev_right_pupil
                            x = int((1 - self.detection_params['smoothing_factor']) * prev_x + self.detection_params['smoothing_factor'] * x)
                            y = int((1 - self.detection_params['smoothing_factor']) * prev_y + self.detection_params['smoothing_factor'] * y)
                        self.prev_right_pupil = (x, y, radius)
                    
                    # Draw pupil
                    cv2.circle(eye_roi_color, (int(x), int(y)), int(radius), (0, 255, 0), 2)

                    # Calculate eye center using landmarks
                    eye_center_x = (eye_landmarks[0][0] + eye_landmarks[3][0]) / 2 - x_min
                    eye_center_y = (eye_landmarks[1][1] + eye_landmarks[4][1]) / 2 - y_min

                    # Calculate vector from eye center to pupil center
                    gaze_vector = np.array([x - eye_center_x, y - eye_center_y])

                    # Normalize the gaze vector
                    norm = np.linalg.norm(gaze_vector)
                    if norm != 0:
                        gaze_vector_normalized = gaze_vector / norm
                    else:
                        gaze_vector_normalized = gaze_vector

                    # Update staring state for current eye
                    if gaze_vector_normalized[0] > self.detection_params['staring_threshold']:
                        self.staring_state["left" if i == 0 else "right"] = True
                    else:
                        self.staring_state["left" if i == 0 else "right"] = False

                    # Determine text and color based on current eye's staring state
                    staring_text = "Staring" if self.staring_state["left" if i == 0 else "right"] else "Not Staring"
                    text_color = (0, 255, 0) if self.staring_state["left" if i == 0 else "right"] else (0, 0, 255)

                    cv2.putText(frame, staring_text, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                
                # Draw eye region
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize pupil detector
    detector = PupilDetector()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = detector.detect_gaze(frame)
        
        # Display
        cv2.imshow('Advanced Pupil Detection', processed_frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()