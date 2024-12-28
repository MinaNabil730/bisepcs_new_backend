import cv2
import mediapipe as mp
import numpy as np
import time

class ExerciseTracker:
    def __init__(self, target_reps=10, target_sets=3, rest_duration=30):
        # MediaPipe Pose Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Exercise Parameters
        self.target_reps = target_reps
        self.target_sets = target_sets
        self.rest_duration = rest_duration  # Rest duration between sets
        
        # Counters
        self.right_reps = 0
        self.left_reps = 0
        self.sets = 0

        # State Tracking
        self.right_hand = True
        self.push_up_right = False
        self.push_up_left = False
        
        # Thresholds
        self.up_thresh = 160  # Full extension
        self.down_thresh = 90  # Full flexion

        # Instruction and Feedback
        self.Instruction_c1 = "Start with right arm"
        self.Instruction_c2 = "Correct Posture"
        
        # Workout State
        self.workout_complete = False
        self.is_resting = False
        self.rest_start_time = None

        # Posture Check Variables
        self.incorrect_posture = False
        self.bad_pos = 0

        # Arm Status Tracking
        self.left_status = "Relaxed"
        self.right_status = "Relaxed"

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def get_angle2(self, A, B):
        """Calculate angle between two points relative to the ground."""
        x1, y1 = A[0], A[1]
        x2, y2 = B[0], B[1]
        angle = (180 + np.degrees(np.arctan2((y1 - y2), (x1 - x2)))) % 180
        return angle

    def handle_posture_check(self, b_angle, ra_angle, la_angle):
        """Handle posture checking logic"""
        self.incorrect_posture = False
        if self.right_hand:
            if abs(b_angle - ra_angle) >= 10:
                if self.bad_pos > 5:
                    if "Please stand straight" in self.Instruction_c2:
                        self.incorrect_posture = True
                    else:
                        self.bad_pos = 0
                else:
                    self.bad_pos += 1
            else:
                self.Instruction_c2 = "Correct Posture"
                self.bad_pos = 0
        else:
            if abs(b_angle - la_angle) >= 10:
                if self.bad_pos > 5:
                    if "Please stand straight" in self.Instruction_c2:
                        self.incorrect_posture = True
                    else:
                        self.bad_pos = 0
                else:
                    self.bad_pos += 1
            else:
                self.Instruction_c2 = "Correct Posture"
                self.bad_pos = 0

        # Check body posture
        if abs(90 - b_angle) >= 10:
            self.Instruction_c2 = "Please stand straight"
            self.incorrect_posture = True

    def process_frame(self, img):
        """Process the frame and track exercise."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.workout_complete:
            return img

        if self.is_resting:
            # Show rest countdown
            if self.rest_start_time is None:
                self.rest_start_time = time.time()

            elapsed_time = time.time() - self.rest_start_time
            remaining_time = int(self.rest_duration - elapsed_time)
            if remaining_time > 0:
                cv2.putText(img, f'Resting... {remaining_time}s remaining', (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.is_resting = False
                self.rest_start_time = None
                self.Instruction_c1 = "Start with right arm"
                
            return img

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints for angle calculation
            left_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            left_elbow = (landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
            left_wrist = (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y)
            
            right_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            right_elbow = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
            right_wrist = (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y)

            # Body and position angles
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, 
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Calculate body angle and arm angles
            b_angle = self.get_angle2(nose, hip)
            ra_angle = self.get_angle2(right_shoulder, right_elbow)
            la_angle = self.get_angle2(left_shoulder, left_elbow)

            # Posture Check
            self.handle_posture_check(b_angle, ra_angle, la_angle)

            # Calculate arm angles
            left_hand_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_hand_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Display posture warnings if incorrect
            if self.incorrect_posture:
                cv2.putText(img, "INCORRECT POSTURE", (20, 370), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, self.Instruction_c2, (20, 410), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return img


            # Check flexed or relaxed status
            self.left_status = "Flex" if left_hand_angle > self.up_thresh else "Relax"
            self.right_status = "Flex" if right_hand_angle > self.up_thresh else "Relax"


            # Tracking reps for the current active arm
            if self.right_hand:
                if right_hand_angle < self.down_thresh and not self.push_up_right:
                    self.push_up_right = True
                elif right_hand_angle > self.up_thresh and self.push_up_right:
                    self.right_reps += 1
                    self.push_up_right = False

                # Switch to left arm when right arm completes reps
                if self.right_reps >= self.target_reps:
                    self.right_hand = False
                    self.Instruction_c1 = "Switch to left arm"
            else:
                if left_hand_angle < self.down_thresh and not self.push_up_left:
                    self.push_up_left = True
                elif left_hand_angle > self.up_thresh and self.push_up_left:
                    self.left_reps += 1
                    self.push_up_left = False

                # Complete a set when left arm finishes reps
                if self.left_reps >= self.target_reps:
                    self.sets += 1
                    
                    # Check if all sets are complete
                    if self.sets >= self.target_sets:
                        self.workout_complete = True
                        self.Instruction_c1 = "Workout Complete!"
                    else:
                        # Start rest period
                        self.is_resting = True
                        self.right_hand = True
                        self.right_reps = 0
                        self.left_reps = 0
                        self.Instruction_c1 = "Resting... Start next set after rest"

        return img

    def get_exercise_state(self):
        """Return the current state of the exercise."""
        return {
            "right_reps": self.right_reps,
            "left_reps": self.left_reps,
            "sets": self.sets,
            "target_reps": self.target_reps,
            "target_sets": self.target_sets,
            "workout_complete": self.workout_complete,
            "current_instruction": self.Instruction_c1,
            "posture_instruction": self.Instruction_c2,
            "resting": self.is_resting,
            "left_status": self.left_status,
            "right_status": self.right_status
        }