#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from exam_interfaces.srv import CheckViolation
from exam_interfaces.msg import DetectionList, FaceData, FaceList
import os
import json

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        self.timer = self.create_timer(1, self.display_status)

        # Subscribers
        self.create_subscription(FaceList, '/face_data', self.face_cb, 10)
        self.create_subscription(DetectionList, '/object_data', self.obj_cb, 10)
        self.create_subscription(Image, '/depth_data', self.depth_cb, 10)
        self.create_subscription(String, '/behavior_state', self.behavior_cb, 10)
        self.create_subscription(String, '/violation_event', self.violation_cb, 10)
        self.create_subscription(String, '/alert_status', self.alert_cb, 10)

        # Service Client
        self.client = self.create_client(CheckViolation, '/check_violation')

        self.bridge = CvBridge()

        # dictionary for topic messages
        self.data_log = {
            "faces": "N/A",
            "object": "N/A", 
            "depth": 0.0,
            "distance": "appropriate",
            "behavior": "Normal", 
            "violation_details": "Student is focused", 
            "alert": "Idle",
            "service_msg": "No violations yet",
            "violation_detected": False,
        }

    # callback functions for subscribers
    def face_cb(self, msg):
        num_faces = len(msg.detections)
        self.data_log["faces"] = f"{num_faces} faces detected"         

    def obj_cb(self, msg):
        detected_list = [d.class_name for d in msg.detections]
        
        if detected_list:
            self.data_log["object"] = ", ".join(detected_list)
        else:
            self.data_log["object"] = "None"

    def depth_cb(self, msg):
        try:
            depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            height, width = depth_map.shape
            center_dist = float(depth_map[height//2, width//2])
            self.data_log["depth"] = center_dist
        except Exception as e:
            self.get_logger().error(f"Monitor Depth Error: {e}")

        if self.data_log["depth"] < 19.2 :
            self.data_log["distance"] = "too far"
        elif self.data_log["depth"] > 19.8:
            self.data_log["distance"] = "too close"
        else:
            self.data_log["distance"] = "appropriate"


    def behavior_cb(self, msg):
        try:
            state_data = json.loads(msg.data)
            self.data_log["behavior"] = state_data.get("status", "N/A")
        except:
            self.get_logger().error("Failed to parse behavior JSON")


    def violation_cb(self, msg):
        try:
            data = json.loads(msg.data)
            details = data.get("details", [])
            
            if details:
                self.data_log["violation_details"] = " | ".join(details)
            else:
                self.data_log["violation_details"] = "None"
        except json.JSONDecodeError:

            self.data_log["violation_details"] = msg.data


    def alert_cb(self, msg): self.data_log["alert"] = msg.data



    def display_status(self):
        os.system('cls' if os.name == 'nt' else 'clear') # clear terminal

        if self.data_log['behavior'] == "Normal":
            self.data_log['violation_details'] = "Student is focused"

        if self.client.service_is_ready():
            request = CheckViolation.Request()
            future = self.client.call_async(request)
            future.add_done_callback(self.service_response_callback)

        print("=" * 30)
        print("===  EXAM PROCTOR MONITOR  ===")
        print("=" * 30)
        
        print(f"Faces:           {self.data_log['faces']}")
        print(f"Object Detected: {self.data_log['object']}")
        print(f"Distance:        {self.data_log['distance']}")
        print(f"Behavior State:  {self.data_log['behavior']}")
        print(f"Violation Event: {self.data_log['violation_details']}")
        print(f"Alert Status:    {self.data_log['alert']}")
        
        print("")
        print("-"*30)
        print("Violation Check: ")
        print(f"{self.data_log['service_msg']}")

    def service_response_callback(self, future):
        try:
            response = future.result()
            if response.message: # Only update if there's actually a string
                self.data_log['service_msg'] = response.message
                self.data_log['violation_detected'] = response.violation_detected
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()