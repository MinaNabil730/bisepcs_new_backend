import cv2
import mediapipe as mp
import time
from math import acos, degrees, sqrt, atan
import base64

class ExerciseTracker:
    def __init__(self, target_reps=10, target_sets=3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.target_reps = target_reps
        self.target_sets = target_sets
        self.right_reps = 0
        self.left_reps = 0
        self.sets = 0
        self.right_hand = True  # Start with right arm
        self.Instruction_c1 = "Start with right arm"
        self.Instruction_c2 = ""
        self.rest_start_time = None
        self.rest_duration = 30  # 30 seconds rest between sets
        self.is_resting = False
        self.workout_complete = False
        self.arm_flexed = False  # Tracks if arm is in flexed state
        self.arm_extended = False  # Tracks if arm is in extended state
        self.angle_threshold_up = 160  # Extension threshold
        self.angle_threshold_down = 90  # Flexion threshold
        self.bad_pos = 0  # Counter for bad arm position
        self.start_time = None
        self.incorrect_posture = False  # Flag for incorrect posture
        self.initial_instruction_given = False  # Flag for initial instruction
        # Add new state variables for tracking rep states
        self.current_state = "waiting"  # Can be "waiting", "flexed", or "extended"
        self.last_angle = None

    def get_angle(self, A, B, C):
        """Calculate the angle between three points (A, B, C)."""
        if 0 not in [A[2], B[2], C[2]]:
            AB = sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
            AC = sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)
            BC = sqrt((C[0] - B[0]) ** 2 + (C[1] - B[1]) ** 2)
            angle = degrees(acos((AB ** 2 + BC ** 2 - AC ** 2) / (2 * AB * BC)))
            return angle
        else:
            return 0

    def get_angle2(self, A, B):
        """Calculate angle between two points relative to the ground."""
        if 0 not in [A[2], B[2]]:
            x1, y1 = A[0], A[1]
            x2, y2 = B[0], B[1]
            angle = (180 + degrees(atan((y1 - y2) / (x1 - x2)))) % 180
            return angle
        else:
            return 0

    def get_feedback(self, angle):
        """Provide feedback on arm position based on the angle."""
        if angle < self.angle_threshold_down:
            return "relax", "#FFDDDD"
        elif angle > self.angle_threshold_up:
            return "flex", "#DDFFDD"
        else:
            return "", "#FFFFFF"

    def count_rep(self, current_angle):
        """Improved rep counting logic with state machine approach"""
        if self.last_angle is None:
            self.last_angle = current_angle
            return

        if self.current_state == "waiting":
            if current_angle < self.angle_threshold_down:
                self.current_state = "flexed"
                # self.Instruction_c1 = "Good flexion, now extend"
        elif self.current_state == "flexed":
            if current_angle > self.angle_threshold_up:
                self.current_state = "extended"
                if self.right_hand:
                    self.right_reps += 1
                else:
                    self.left_reps += 1
                self.current_state = "waiting"  # Reset for next rep
                # self.Instruction_c1 = "Good extension, now flex"

        self.last_angle = current_angle

    def handle_set_completion(self):
        """Handle the completion of a set"""
        self.sets += 1
        if self.sets < self.target_sets:
            self.rest_start_time = time.time()
            self.is_resting = True
            self.Instruction_c1 = f"Completed Set {self.sets}. Rest for 30 seconds."
            self.right_hand = True  # Reset to right hand for next set
            self.Instruction_c2 = "Next set: Right arm turn"
            self.right_reps = 0  # Reset reps for both arms
            self.left_reps = 0
            # Reset states
            self.current_state = "waiting"
            self.last_angle = None
        else:
            self.workout_complete = True
            self.Instruction_c1 = "Great job! You've completed your workout!"
            self.Instruction_c2 = "Keep up the good work!"

    def handle_rest_period(self):
        """Handle the rest period between sets"""
        elapsed_time = time.time() - self.rest_start_time
        if elapsed_time < self.rest_duration:
            self.Instruction_c2 = f"Rest: {int(self.rest_duration - elapsed_time)} seconds left"
        else:
            self.is_resting = False
            self.rest_start_time = None
            self.Instruction_c1 = f"Start Set {self.sets + 1} with your right arm"
            self.Instruction_c2 = "Right arm"
            # Reset states for new set
            self.current_state = "waiting"
            self.last_angle = None

    def handle_posture_check(self, b_angle, ra_angle, la_angle):
        """Handle posture checking logic"""
        self.incorrect_posture = False
        if self.right_hand:
            if 0 not in [b_angle, ra_angle] and abs(b_angle - ra_angle) >= 10:
                if self.bad_pos > 5:
                    self.incorrect_posture = True
                else:
                    self.bad_pos += 1
            else:
                self.Instruction_c2 = ""
                self.bad_pos = 0
        else:
            if 0 not in [b_angle, la_angle] and abs(b_angle - la_angle) >= 10:
                if self.bad_pos > 5:
                    self.incorrect_posture = True
                else:
                    self.bad_pos += 1
            else:
                self.Instruction_c2 = ""
                self.bad_pos = 0

        # Check body posture
        if abs(90 - b_angle) >= 10:
            self.Instruction_c2 = "Please stand straight"
            self.incorrect_posture = True

    def process_frame(self, img):
        """Process the frame from the camera, calculate arm angles, and count reps."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.start_time is None:
            self.start_time = time.time()

        if not self.initial_instruction_given:
            self.Instruction_c1 = "Start with right arm"
            self.initial_instruction_given = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Right arm keypoints
            r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility]
            r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility]
            r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility]

            # Left arm keypoints
            l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility]
            l_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility]
            l_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].visibility]

            # Body keypoints
            b_top = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].y,
                    landmarks[self.mp_pose.PoseLandmark.NOSE.value].visibility]
            b_bottom = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility]

            # Calculate arm angles
            r_angle = self.get_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = self.get_angle(l_shoulder, l_elbow, l_wrist)

            # Calculate body and upper arm angles
            b_angle = self.get_angle2(b_top, b_bottom)
            ra_angle = self.get_angle2(r_shoulder, r_elbow)
            la_angle = self.get_angle2(l_shoulder, l_elbow)

            # Determine which arm we are working on
            if not self.is_resting and not self.workout_complete:
                current_angle = r_angle if self.right_hand else l_angle
                feedback, color = self.get_feedback(current_angle)
                self.Instruction_c1 = feedback

                # Handle posture checks
                self.handle_posture_check(b_angle, ra_angle, la_angle)

                # Only count reps if posture is correct
                if not self.incorrect_posture:
                    self.count_rep(current_angle)

                # Check if the right arm has completed its reps
                if self.right_reps >= self.target_reps and self.right_hand:
                    self.right_hand = False  # Switch to left arm
                    self.Instruction_c1 = "Switch to left arm" 
                    # Reset states for left arm
                    self.current_state = "waiting"
                    self.last_angle = None

                # Check if both arms have completed their reps
                if self.left_reps >= self.target_reps and not self.right_hand:
                    self.handle_set_completion()

            if self.is_resting:
                self.handle_rest_period()

        return img

    def get_exercise_state(self):
        """Return the current state of the exercise session."""
        return {
            "current_arm": "Right" if self.right_hand else "Left",
            "right_reps": self.right_reps,
            "left_reps": self.left_reps,
            "sets": self.sets,
            "target_reps": self.target_reps,
            "target_sets": self.target_sets,
            "is_resting": self.is_resting,
            "workout_complete": self.workout_complete,
            "instruction_1": self.Instruction_c1,
            "instruction_2": self.Instruction_c2,
        }