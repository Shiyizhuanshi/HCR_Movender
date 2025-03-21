import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
import sys

# Mediapipe相关初始化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 封装了人脸、手势、姿势等检测逻辑的类
class allin1(mp_holistic.Holistic):
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 model_complexity=0, refine_face_landmarks=True):
        super().__init__(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            refine_face_landmarks=refine_face_landmarks
        )
        # 新的情绪映射：Surprised: +2, Happy: +1, Neutral: 0, Disgusted: -1
        self.emo_dict = {
            "Surprised": 2,
            "Happy": 1,
            "Neutral": 0,
            "Disgusted": -1
        }
        # 新的手势映射：Indexing: +1.5, Hi: +1, Thumb up: +0.5, Rejecting: -1, Nothing: 0
        self.hand_dict = {
            "Thumb up": 0.5,
            "Indexing": 1.5,
            "Hi": 1,
            "Rejecting": -1,
            "Nothing": 0
        }
        # 不再使用原来的姿态字典，而是直接根据pose_ratio计算评分
        self.emotion = "Neutral"
        self.gesture = "Nothing"
        self.pose_score = 0
        self.emo_score = 0
        self.gesture_score = 0
        # 新增三个额外评分
        self.gaze_score = 0
        self.head_score = 0
        self.hand_movement_score = 0

        self.center_height = 0
        self.lookaway = False

        # 用于手部运动检测（记录前一帧手部位置，简单取水平位置）
        self.prev_hand_x = None

        # 保持原有的手势识别辅助列表（索引顺序对应：Thumb up, Indexing, Hi）
        self.hand_list = ["Thumb up", "Indexing", "Hi"]

    def distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def sqDistance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def predict_emotion(self, face_landmarks):
        # 根据新的算法：Surprised、Happy、Neutral、Disgusted
        self.emotion = "Neutral"
        if face_landmarks:
            landmarks = face_landmarks.landmark
            # 计算一些关键比例（仍然使用部分原有逻辑）
            forehead_size_left = self.sqDistance(landmarks[332], landmarks[282])
            forehead_size_right = self.sqDistance(landmarks[103], landmarks[52])
            mouth_openness = self.sqDistance(landmarks[13], landmarks[14])
            left_brow_raise = self.sqDistance(landmarks[52], landmarks[282])
            right_brow_raise = self.sqDistance(landmarks[282], landmarks[472])
            mouth_width = self.sqDistance(landmarks[61], landmarks[291])
            face_height = self.sqDistance(landmarks[10], landmarks[1])

            mouth_open_ratio = mouth_openness / face_height if face_height > 0 else 0
            smile_ratio = mouth_width / mouth_openness if mouth_openness > 0 else 0
            brow_raise_ratio = (left_brow_raise / forehead_size_left + right_brow_raise / forehead_size_right) if (forehead_size_left > 0 and forehead_size_right > 0) else 0

            # 如果检测到“看向别处”，则视为负面表达
            if self.lookaway:
                self.emotion = "Disgusted"
            # 条件判断顺序：首先判断惊讶，再判断快乐，再默认为中性
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
        self.gesture = "Nothing"
        distances0 = []
        distances1 = []
        # 同时检测左右手
        if landmarks_right and landmarks_left:
            distances0 = [self.sqDistance(landmarks_right.landmark[i], landmarks_right.landmark[0]) for i in [4, 8, 12, 16, 20]]
            distances1 = [self.sqDistance(landmarks_left.landmark[i], landmarks_left.landmark[0]) for i in [4, 8, 12, 16, 20]]
            # 归一化
            norm0 = max(distances0) if max(distances0) != 0 else 1
            norm1 = max(distances1) if max(distances1) != 0 else 1
            distances0 = [d / norm0 for d in distances0]
            distances1 = [d / norm1 for d in distances1]

            # 根据距离总和判断拒绝（Rejecting）手势
            if (sum(distances0) > 4.3 and sum(distances1) > 4.3 and 
                landmarks_right.landmark[0].y < self.center_height and 
                landmarks_left.landmark[0].y < self.center_height):
                self.gesture = "Rejecting"
            # 根据单侧手势判定打招呼（Hi）
            elif ((sum(distances0) > 4.4 and landmarks_left.landmark[0].y < self.center_height) or 
                  (sum(distances1) > 4.4 and landmarks_right.landmark[0].y < self.center_height)):
                self.gesture = "Hi"
            # 如果某一侧的最大距离点索引为0，则判定为Thumb up
            elif np.argmax(distances0) == 0 or np.argmax(distances1) == 0:
                self.gesture = "Thumb up"
            else:
                # 若没有明显其他手势，优先选择 Indexing（主动选择）
                self.gesture = "Indexing"
        elif landmarks_right:
            distances = [self.sqDistance(landmarks_right.landmark[i], landmarks_right.landmark[0]) for i in [4, 8, 12, 16, 20]]
            self.gesture = self.hand_list[min(np.argmax(distances), 2)]
        elif landmarks_left:
            distances = [self.sqDistance(landmarks_left.landmark[i], landmarks_left.landmark[0]) for i in [4, 8, 12, 16, 20]]
            self.gesture = self.hand_list[min(np.argmax(distances), 2)]

        self.gesture_score = self.hand_dict[self.gesture]
    
    def predict_pose(self, pose_landmarks):
        # 计算姿态评分基于pose ratio
        if pose_landmarks:
            landmarks = pose_landmarks.landmark
            # 计算三个距离：此处使用关键点6、3、0作为示例
            d1 = self.distance(landmarks[6], landmarks[3])
            d2 = self.distance(landmarks[6], landmarks[0])
            d3 = self.distance(landmarks[3], landmarks[0])
            # 计算pose_ratio
            if (d2 + d3) > 0:
                pose_ratio = 2 * d1 / (d2 + d3)
            else:
                pose_ratio = 1  # 默认中性值

            # 更新lookaway标记，用于情绪判断（可根据实际情况调整阈值）
            self.lookaway = (pose_ratio < 1.3)

            # 根据pose_ratio计算姿态评分：>1.5为开放(+2)，0.8~1.5为中性(0)，<0.8为封闭(-2)
            if pose_ratio > 1.5:
                self.pose_score = 2
            elif 0.8 <= pose_ratio <= 1.5:
                self.pose_score = 0
            else:
                self.pose_score = -2

            # 更新中心高度（用于后续手势判断）
            self.center_height = (landmarks[12].y + landmarks[24].y + landmarks[11].y + landmarks[23].y) / 4
        else:
            self.pose_score = 0

    def predict_gaze(self, face_landmarks):
        # 简单判断面部中心是否在画面水平中间
        if face_landmarks:
            nose = face_landmarks.landmark[1]  # 近似取第2个关键点作为鼻子
            if 0.4 < nose.x < 0.6:
                return 1  # 注视售货机
        return 0

    def predict_head_orientation(self, face_landmarks):
        # 使用左右眼关键点判断头部是否正对售货机（简单示例）
        if face_landmarks:
            left_eye = face_landmarks.landmark[33]   # 近似左眼
            right_eye = face_landmarks.landmark[263]   # 近似右眼
            # 如果两眼垂直位置差异较小，认为头部正对；若差异较大，认为偏离
            if abs(left_eye.y - right_eye.y) < 0.02:
                return 1   # 正对
            elif left_eye.y - right_eye.y > 0.02:
                return 0   # 中性
            else:
                return -1  # 偏离
        return 0

    def update_hand_movement(self, landmarks_right, landmarks_left):
        # 根据当前帧的手部水平位置与上一帧比较，判断手势是否接近售货机
        current_hand_x = None
        if landmarks_right:
            current_hand_x = landmarks_right.landmark[0].x
        elif landmarks_left:
            current_hand_x = landmarks_left.landmark[0].x
        score = 0
        if current_hand_x is not None:
            if self.prev_hand_x is not None:
                if current_hand_x < self.prev_hand_x:
                    score = 1   # 手部靠近（假设x值减小代表靠近）
                elif current_hand_x > self.prev_hand_x:
                    score = -1  # 手部远离
                else:
                    score = 0
            self.prev_hand_x = current_hand_x
        return score

    def detect(self, frame, close_detect):
        # 将frame设为只读进行处理
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.process(frame_rgb)
        # 初始化各项评分
        self.emo_score = 0
        self.gesture_score = 0
        self.pose_score = 0
        self.gaze_score = 0
        self.head_score = 0
        self.hand_movement_score = 0

        # 恢复写权限（后续绘制逻辑均已注释）
        frame.flags.writeable = True

        # 姿态检测（始终执行）
        self.predict_pose(results.pose_landmarks)
        if close_detect:
            # 仅在近距离时执行面部和手势检测
            self.predict_emotion(results.face_landmarks)
            self.predict_hand(results.right_hand_landmarks, results.left_hand_landmarks)
            self.gaze_score = self.predict_gaze(results.face_landmarks) if results.face_landmarks else 0
            self.head_score = self.predict_head_orientation(results.face_landmarks) if results.face_landmarks else 0
            self.hand_movement_score = self.update_hand_movement(results.right_hand_landmarks, results.left_hand_landmarks)
        # 计算总体 CV 分数：各部分评分之和
        total_cv_score = (self.emo_score + self.pose_score + self.gesture_score +
                          self.gaze_score + self.head_score + self.hand_movement_score)
        return frame, total_cv_score


# 封装整个检测流程为一个可直接调用的类
class CVGoalDetector:
    def __init__(self, state, display=False, threshold=1.5, tolerance=0.2):
        """
        :param state: 一个字典，用于存储和共享检测状态信息
        :param display: 是否显示视频（本例中已将视频输出逻辑注释掉）
        :param threshold: 距离阈值（单位：米），低于此距离认为“靠近”
        :param tolerance: 在计算平均score时，允许的距离容差（单位：米），仅对与最近距离差值小于该容差的目标有效
        """
        self.display = display
        self.threshold = threshold
        self.tolerance = tolerance
        self.state = state
        self.engaged = False  # 标记是否已经触发过"开始对话"信号
        self.goal_pos = None  # 最近目标的坐标
        self.allin1_detector = allin1()

        # 模型文件路径，注意修改为你自己的路径
        self.nnPath = str((Path(__file__).parent / Path('models/yolov8n_coco_640x352.blob')).resolve().absolute())
        if not Path(self.nnPath).exists():
            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        # 构造DepthAI的pipeline
        self.pipeline = dai.Pipeline()
        self._setup_pipeline()

    def _setup_pipeline(self):
        # 创建RGB摄像头节点
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 352)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

        # 创建深度节点
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

        # 创建YOLO检测网络节点
        detectionNetwork = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.setNumClasses(80)
        detectionNetwork.setCoordinateSize(4)
        detectionNetwork.setIouThreshold(0.5)
        detectionNetwork.setBlobPath(self.nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)
        detectionNetwork.setBoundingBoxScaleFactor(0.5)
        detectionNetwork.setDepthLowerThreshold(100)   # 毫米
        detectionNetwork.setDepthUpperThreshold(100000)  # 毫米

        stereo.depth.link(detectionNetwork.inputDepth)
        camRgb.preview.link(detectionNetwork.input)

        # 创建输出节点（只获取数据，不显示视频）
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        detectionNetwork.passthrough.link(xoutRgb.input)

        nnOut = self.pipeline.create(dai.node.XLinkOut)
        nnOut.setStreamName("nn")
        detectionNetwork.out.link(nnOut.input)

    def run(self):
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
                    detections = [det for det in detections if det.label == 0]  # 仅检测"person"

                persons_coords = []
                scores = []
                cv_score = 0

                for det in detections:
                    # 毫米转为米
                    x_m = det.spatialCoordinates.x / 1000.0
                    z_m = det.spatialCoordinates.z / 1000.0
                    # 根据新算法计算距离评分：d = sqrt(x^2+z^2)，S_distance = 2/d
                    d = np.sqrt(x_m**2 + z_m**2)
                    dist_score = 2 / np.maximum(d, 0.001)
                    persons_coords.append((x_m, z_m))
                    bbox = [int(det.xmin * 640), int(det.ymin * 352), int(det.xmax * 640), int(det.ymax * 352)]
                    if frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].size != 0:
                        # 当距离小于1.5米时启用面部、手势检测
                        _, cv_score = self.allin1_detector.detect(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], d < 1.5)
                    # 总体得分 = 距离评分 + CV检测各模块得分
                    total_score = dist_score + cv_score
                    # 将得分取整：四舍五入到最近的整数
                    total_score_int = int(round(total_score))
                    scores.append(total_score_int)

                if len(persons_coords) != 0:
                    # 使用最近的人的坐标，即z值最小的那个人
                    closest_index = np.argmin([coord[1] for coord in persons_coords])
                    closest_person_coord = persons_coords[closest_index]
                    closest_distance = persons_coords[closest_index][1]
                    # 更新目标坐标为最近的人的坐标
                    self.goal_pos = closest_person_coord

                    # engaged 状态：只要最近的人的距离小于阈值则为True，否则False
                    self.engaged = closest_distance < self.threshold

                    # 计算平均 score：
                    # 取所有与最近距离差值在 tolerance 范围内且得分在 -2 到 8 之间的目标
                    valid_scores = []
                    for i, coord in enumerate(persons_coords):
                        if abs(coord[1] - closest_distance) < self.tolerance and (-2 <= scores[i] <= 8):
                            valid_scores.append(scores[i])
                    if valid_scores:
                        avg_score = int(round(sum(valid_scores) / len(valid_scores)))
                    else:
                        avg_score = None

                    self.state["engaged"] = self.engaged
                    self.state["goal_pos"] = self.goal_pos
                    self.state["score"] = avg_score

                    # 打印状态信息
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
    # 传入共享状态字典
    state = {"engaged": False, "goal_pos": None, "score": None}
    detector = CVGoalDetector(state=state, display=False)
    detector.run()
