import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from exam_interfaces.msg import FaceList, DetectionList
from sensor_msgs.msg import Image
from std_msgs.msg import String
import json

class BehaviorAnalysisNode(Node):
    def __init__(self):
        super().__init__('behavior_analysis_node')
        self.get_logger().info("Behavior Analysis Node Started.")

        self.declare_parameter("attention_threshold", 1.5)

        self.face_sub     = self.create_subscription(FaceList, '/face_data', self.face_callback, 10)
        self.object_sub   = self.create_subscription(DetectionList, '/object_data', self.object_callback, 10)
        self.depth_sub    = self.create_subscription(Image, '/depth_data', self.depth_callback, 10)

        self.behavior_pub = self.create_publisher(String, '/behavior_state', 10) 

        self.current_face   = "looking_forward"
        self.current_object = "none"
        self.current_depth  =  0.5

        self.object_dir = {}
        self.face_dir = {}
        self.depth_dir = {}
        
        self.face_counts = 0
        self.face_detected = False

        # Camera Intrinsics
        FX = 500.0  
        FY = 500.0
        CX = 320.0
        CY = 240.0

        self.bridge = CvBridge()

        self.timer = self.create_timer((0.2), self.analyze_behavior)

    def object_callback(self, object_msg):
        self.object_dir[object_msg.header.frame_id] = object_msg.detections

    def face_callback(self, face_msg):
        self.face_dir[face_msg.header.frame_id] = face_msg.detections

    def depth_callback(self,depth_msg):
        depth_map = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        self.depth_dir[depth_msg.header.frame_id] = depth_map

    def process(self):
        if len(self.object_dir) == 0 or len(self.face_dir) == 0 or len(self.depth_dir) == 0 :
            self.get_logger().info(f"Waiting for data....")
            return

        common_ids = set(self.object_dir.keys()) & set(self.face_dir.keys()) & set(self.depth_dir.keys())
        if not common_ids:
            self.get_logger().warn("No matching Frame IDs found yet....")
            return None
        
        last_sync_id = max(common_ids, key=int)

        sync_objects = self.object_dir[last_sync_id]
        sync_faces   = self.face_dir[last_sync_id]
        sync_depth   = self.depth_dir[last_sync_id]

        return sync_objects, sync_faces, sync_depth


    def analyze_behavior(self):
        sync_data = self.process()
        if sync_data is None:
            return
        
        # objs = List of BoundingBox
        # faces = List of FaceData
        # depth_map = numpy array (32FC1)
        objs, faces, depth_map = sync_data

        attention_threshold = self.get_parameter('attention_threshold').get_parameter_value().double_value

        behavior_issues = []

        # --- Face Logic ---
        if len(faces) == 0:
            behavior_issues.append("Violation: No Face Detected!")
        else:
            count = faces[0].face_count 
            if count == 0:
                behavior_issues.append("Violation: No Face Detected!")
            elif count > 1:
                behavior_issues.append("Violation: Multiple People Detected!")

        # --- Object Logic ---
        for detection in objs:
            if detection.class_name in ["cell phone", "book"]:
                behavior_issues.append(f"Prohibited Object Detected: {detection.class_name}")

        # --- Depth Logic ---
        height, width = depth_map.shape
        center_dist = float(depth_map[height//2, width//2])
        
        if center_dist > attention_threshold:
            behavior_issues.append("Student too far (Unusual Distance).")
        elif center_dist < 0.2:
            behavior_issues.append("Student too close (Unusual Distance).")
        
        state_msg = String()

        if len(behavior_issues) > 0:
            state_data = {
                "status": "Suspicious",
                "reasons": behavior_issues
            }
        else:
            state_data = {
                "status": "Normal",
                "reasons": ["Student is focused."]
            }
        
        state_msg.data = json.dumps(state_data)
        self.behavior_pub.publish(state_msg)
        self.get_logger().debug(f"Published Behavior State: {state_data['status']}")
            
def main(args=None):
    rclpy.init(args=args)
    node = BehaviorAnalysisNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
