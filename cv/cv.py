import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


class allin1(mp_holistic.Holistic):
    """
    A holistic detection class that extends Mediapipe's Holistic solution.

    This class encapsulates the logic for face, hand, and pose detection,
    and computes various scores based on the detection results.

    Attributes:
        emo_dict (dict): Mapping of emotions to scores.
        hand_dict (dict): Mapping of hand gestures to scores.
        emotion (str): Current detected emotion.
        gesture (str): Current detected hand gesture.
        pose_score (int): Score based on pose detection.
        emo_score (int): Score based on emotion detection.
        gesture_score (int): Score based on hand gesture detection.
        gaze_score (int): Score based on gaze detection.
        head_score (int): Score based on head orientation detection.
        hand_movement_score (int): Score based on hand movement.
        center_height (float): Center height used for gesture decisions.
        lookaway (bool): Flag indicating if the subject is looking away.
        prev_hand_x (float): Previous horizontal hand position for movement detection.
        hand_list (list): List of hand gesture names used for decision making.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 model_complexity=0, refine_face_landmarks=True):
        """
        Initialize the allin1 detector with specified detection parameters.

        :param min_detection_confidence: Minimum confidence value for detection.
        :param min_tracking_confidence: Minimum confidence value for tracking.
        :param model_complexity: Complexity of the model.
        :param refine_face_landmarks: Whether to refine face landmarks.
        """
        super().__init__(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks
        )
        self.emo_dict = {
            "Surprised": 2,
            "Happy": 1,
            "Neutral": 0,
            "Disgusted": -1
        }
        self.hand_dict = {
            "Thumb up": 0.5,
            "Indexing": 1.5,
            "Hi": 1,
            "Rejecting": -1,
            "Nothing": 0
        }
        self.emotion = "Neutral"
        self.gesture = "Nothing"
        self.pose_score = 0
        self.emo_score = 0
        self.gesture_score = 0
        self.gaze_score = 0
        self.head_score = 0
        self.hand_movement_score = 0
        self.center_height = 0
        self.lookaway = False
        self.prev_hand_x = None
        self.hand_list = ["Thumb up", "Indexing", "Hi"]

    def distance(self, p1, p2):
        """
        Compute the Euclidean distance between two 2D points.

        :param p1: First point with attributes x and y.
        :param p2: Second point with attributes x and y.
        :return: Euclidean distance.
        """
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def sqDistance(self, p1, p2):
        """
        Compute the Euclidean distance between two 3D points.

        :param p1: First point with attributes x, y, and z.
        :param p2: Second point with attributes x, y, and z.
        :return: Euclidean distance.
        """
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def predict_emotion(self, face_landmarks):
        """
        Predict emotion based on face landmarks and update the emotion score.

        :param face_landmarks: Face landmarks object from Mediapipe.
        """
        self.emotion = "Neutral"
        if face_landmarks:
            landmarks = face_landmarks.landmark
            forehead_size_left = self.sqDistance(landmarks[332], landmarks[282])
            forehead_size_right = self.sqDistance(landmarks[103], landmarks[52])
            mouth_openness = self.sqDistance(landmarks[13], landmarks[14])
            left_brow_raise = self.sqDistance(landmarks[52], landmarks[282])
            right_brow_raise = self.sqDistance(landmarks[282], landmarks[472])
            mouth_width = self.sqDistance(landmarks[61], landmarks[291])
            face_height = self.sqDistance(landmarks[10], landmarks[1])
            mouth_open_ratio = mouth_openness / face_height if face_height > 0 else 0
            smile_ratio = mouth_width / mouth_openness if mouth_openness > 0 else 0
            brow_raise_ratio = (left_brow_raise / forehead_size_left + right_brow_raise / forehead_size_right) \
                if (forehead_size_left > 0 and forehead_size_right > 0) else 0
            if self.lookaway:
                self.emotion = "Disgusted"
            elif mouth_open_ratio > 0.15 or brow_raise_ratio > 5.9:
                self.emotion = "Surprised"
            elif smile_ratio < 15.0:
                self.emotion = "Happy"
            else:
                self.emotion = "Neutral"
            self.emo_score = self.emo_dict[self.emotion]
        else:
            self.emo_score = self.emo_dict["Neutral"]

    def predict_hand(self, landmarks_right, landmarks_left):
        """
        Predict hand gesture based on the provided hand landmarks and update the gesture score.

        :param landmarks_right: Landmarks for the right hand.
        :param landmarks_left: Landmarks for the left hand.
        """
        self.gesture = "Nothing"
        distances0 = []
        distances1 = []
        if landmarks_right and landmarks_left:
            distances0 = [self.sqDistance(landmarks_right.landmark[i], landmarks_right.landmark[0])
                          for i in [4, 8, 12, 16, 20]]
            distances1 = [self.sqDistance(landmarks_left.landmark[i], landmarks_left.landmark[0])
                          for i in [4, 8, 12, 16, 20]]
            norm0 = max(distances0) if max(distances0) != 0 else 1
            norm1 = max(distances1) if max(distances1) != 0 else 1
            distances0 = [d / norm0 for d in distances0]
            distances1 = [d / norm1 for d in distances1]
            if (sum(distances0) > 4.3 and sum(distances1) > 4.3 and
                    landmarks_right.landmark[0].y < self.center_height and
                    landmarks_left.landmark[0].y < self.center_height):
                self.gesture = "Rejecting"
            elif ((sum(distances0) > 4.4 and landmarks_left.landmark[0].y < self.center_height) or
                  (sum(distances1) > 4.4 and landmarks_right.landmark[0].y < self.center_height)):
                self.gesture = "Hi"
            elif np.argmax(distances0) == 0 or np.argmax(distances1) == 0:
                self.gesture = "Thumb up"
            else:
                self.gesture = "Indexing"
        elif landmarks_right:
            distances = [self.sqDistance(landmarks_right.landmark[i], landmarks_right.landmark[0])
                         for i in [4, 8, 12, 16, 20]]
            self.gesture = self.hand_list[min(np.argmax(distances), 2)]
        elif landmarks_left:
            distances = [self.sqDistance(landmarks_left.landmark[i], landmarks_left.landmark[0])
                         for i in [4, 8, 12, 16, 20]]
            self.gesture = self.hand_list[min(np.argmax(distances), 2)]
        self.gesture_score = self.hand_dict[self.gesture]

    def predict_pose(self, pose_landmarks):
        """
        Predict pose score based on pose landmarks and update center height.

        :param pose_landmarks: Pose landmarks object from Mediapipe.
        """
        if pose_landmarks:
            landmarks = pose_landmarks.landmark
            d1 = self.distance(landmarks[6], landmarks[3])
            d2 = self.distance(landmarks[6], landmarks[0])
            d3 = self.distance(landmarks[3], landmarks[0])
            pose_ratio = 2 * d1 / (d2 + d3) if (d2 + d3) > 0 else 1
            self.lookaway = (pose_ratio < 1.3)
            if pose_ratio > 1.5:
                self.pose_score = 2
            elif 0.8 <= pose_ratio <= 1.5:
                self.pose_score = 0
            else:
                self.pose_score = -2
            self.center_height = (landmarks[12].y + landmarks[24].y + landmarks[11].y + landmarks[23].y) / 4
        else:
            self.pose_score = 0

    def predict_gaze(self, face_landmarks):
        """
        Predict gaze score based on face landmarks.

        :param face_landmarks: Face landmarks object from Mediapipe.
        :return: Gaze score.
        """
        if face_landmarks:
            nose = face_landmarks.landmark[1]
            if 0.4 < nose.x < 0.6:
                return 1
        return 0

    def predict_head_orientation(self, face_landmarks):
        """
        Predict head orientation score based on face landmarks.

        :param face_landmarks: Face landmarks object from Mediapipe.
        :return: Head orientation score.
        """
        if face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            if abs(left_eye.y - right_eye.y) < 0.02:
                return 1
            elif left_eye.y - right_eye.y > 0.02:
                return 0
            else:
                return -1
        return 0

    def update_hand_movement(self, landmarks_right, landmarks_left):
        """
        Update the hand movement score based on horizontal hand movement between frames.

        :param landmarks_right: Landmarks for the right hand.
        :param landmarks_left: Landmarks for the left hand.
        :return: Hand movement score.
        """
        current_hand_x = None
        if landmarks_right:
            current_hand_x = landmarks_right.landmark[0].x
        elif landmarks_left:
            current_hand_x = landmarks_left.landmark[0].x
        score = 0
        if current_hand_x is not None:
            if self.prev_hand_x is not None:
                if current_hand_x < self.prev_hand_x:
                    score = 1
                elif current_hand_x > self.prev_hand_x:
                    score = -1
                else:
                    score = 0
            self.prev_hand_x = current_hand_x
        return score

    def detect(self, frame, close_detect):
        """
        Process an input frame to compute various detection scores.

        :param frame: Input video frame.
        :param close_detect: Flag indicating whether to perform close-range detection.
        :return: Tuple (processed frame, total CV score).
        """
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.process(frame_rgb)
        self.emo_score = 0
        self.gesture_score = 0
        self.pose_score = 0
        self.gaze_score = 0
        self.head_score = 0
        self.hand_movement_score = 0
        frame.flags.writeable = True
        self.predict_pose(results.pose_landmarks)
        if close_detect:
            self.predict_emotion(results.face_landmarks)
            self.predict_hand(results.right_hand_landmarks, results.left_hand_landmarks)
            self.gaze_score = self.predict_gaze(results.face_landmarks) if results.face_landmarks else 0
            self.head_score = self.predict_head_orientation(results.face_landmarks) if results.face_landmarks else 0
            self.hand_movement_score = self.update_hand_movement(results.right_hand_landmarks,
                                                                 results.left_hand_landmarks)
        total_cv_score = (self.emo_score + self.pose_score + self.gesture_score +
                          self.gaze_score + self.head_score + self.hand_movement_score)
        return frame, total_cv_score


class CVGoalDetector:
    """
    A detector that integrates person detection with depth estimation and holistic analysis.

    Combines YOLO-based person detection with DepthAI depth information and Mediapipe holistic detection
    to compute a combined score. It updates a shared state to indicate if a target is engaged.

    Attributes:
        display (bool): Whether to display the video.
        threshold (float): Distance threshold (in meters) for engagement.
        tolerance (float): Tolerance for score averaging (in meters).
        state (dict): Shared state dictionary to store detection status.
        engaged (bool): Flag indicating if the target is engaged.
        goal_pos (tuple): Coordinates of the closest detected person.
        allin1_detector (allin1): Instance of the allin1 detection class.
        nnPath (str): Path to the YOLO model blob.
        pipeline (dai.Pipeline): DepthAI pipeline for detection.
    """

    def __init__(self, state, display=False, threshold=1.5, tolerance=0.2):
        """
        Initialize the CVGoalDetector.

        :param state: Dictionary for shared state.
        :param display: Whether to display the video.
        :param threshold: Distance threshold (in meters) for engagement.
        :param tolerance: Tolerance for score averaging (in meters).
        """
        self.display = display
        self.threshold = threshold
        self.tolerance = tolerance
        self.state = state
        self.engaged = False
        self.goal_pos = None
        self.allin1_detector = allin1()
        self.nnPath = str((Path(__file__).parent / Path('models/yolov8n_coco_640x352.blob')).resolve().absolute())
        if not Path(self.nnPath).exists():
            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')
        self.pipeline = dai.Pipeline()
        self._setup_pipeline()

    def _setup_pipeline(self):
        """
        Set up the DepthAI pipeline with RGB camera, stereo depth, and YOLO spatial detection network.
        """
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 352)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(640, 352)
        left = self.pipeline.create(dai.node.MonoCamera)
        right = self.pipeline.create(dai.node.MonoCamera)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        detectionNetwork = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.setNumClasses(80)
        detectionNetwork.setCoordinateSize(4)
        detectionNetwork.setIouThreshold(0.5)
        detectionNetwork.setBlobPath(self.nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)
        detectionNetwork.setBoundingBoxScaleFactor(0.5)
        detectionNetwork.setDepthLowerThreshold(100)
        detectionNetwork.setDepthUpperThreshold(100000)
        stereo.depth.link(detectionNetwork.inputDepth)
        camRgb.preview.link(detectionNetwork.input)
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        detectionNetwork.passthrough.link(xoutRgb.input)
        nnOut = self.pipeline.create(dai.node.XLinkOut)
        nnOut.setStreamName("nn")
        detectionNetwork.out.link(nnOut.input)

    def run(self):
        """
        Run the main detection loop, process frames, compute scores, and update the shared state.
        """
        with dai.Device(self.pipeline) as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            while True:
                try:
                    inRgb = qRgb.get()
                    inDet = qDet.get()
                except RuntimeError as e:
                    print(f"Error receiving inference data: {e}")
                    continue
                frame = None
                detections = []
                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                if inDet is not None:
                    detections = inDet.detections
                    detections = [det for det in detections if det.label == 0]
                persons_coords = []
                scores = []
                cv_score = 0
                for det in detections:
                    x_m = det.spatialCoordinates.x / 1000.0
                    z_m = det.spatialCoordinates.z / 1000.0
                    d = np.sqrt(x_m ** 2 + z_m ** 2)
                    dist_score = 2 / np.maximum(d, 0.001)
                    persons_coords.append((x_m, z_m))
                    bbox = [int(det.xmin * 640), int(det.ymin * 352), int(det.xmax * 640), int(det.ymax * 352)]
                    if frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].size != 0:
                        _, cv_score = self.allin1_detector.detect(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], d < 1.5)
                    total_score = dist_score + cv_score
                    total_score_int = int(round(total_score))
                    scores.append(total_score_int)
                if len(persons_coords) != 0:
                    closest_index = np.argmin([coord[1] for coord in persons_coords])
                    closest_person_coord = persons_coords[closest_index]
                    closest_distance = persons_coords[closest_index][1]
                    self.goal_pos = closest_person_coord
                    self.engaged = closest_distance < self.threshold
                    valid_scores = []
                    for i, coord in enumerate(persons_coords):
                        if abs(coord[1] - closest_distance) < self.tolerance and (-2 <= scores[i] <= 8):
                            valid_scores.append(scores[i])
                    avg_score = int(round(sum(valid_scores) / len(valid_scores))) if valid_scores else None
                    self.state["engaged"] = self.engaged
                    self.state["goal_pos"] = self.goal_pos
                    self.state["score"] = avg_score
                    print("[CV] Closest person coordinate:", self.goal_pos)
                    print("[CV] Engaged:", self.engaged)
                    print("[CV] Average score:", avg_score)
                else:
                    if self.engaged:
                        print("[CV] No person detected. Reset signal sent.")
                    self.engaged = False
                    self.goal_pos = None
                    self.state["engaged"] = self.engaged
                    self.state["goal_pos"] = None
                    self.state["score"] = None
                time.sleep(0.01)
                if self.display:
                    cv2.imshow("rgb", frame)
                cv2.waitKey(1)


if __name__ == "__main__":
    state = {"engaged": False, "goal_pos": None, "score": None}
    detector = CVGoalDetector(state=state, display=False)
    detector.run()
