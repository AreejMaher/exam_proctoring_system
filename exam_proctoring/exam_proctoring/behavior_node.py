import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import json

class BehaviorAnalysisNode(Node):
    def __init__(self):
        super().__init__('behavior_analysis_node')
        self.get_logger().info("Behavior Analysis Node Started.")

        self.declare_parameter("attention_threshold", 1.5)

        self.face_sub     = self.create_subscription(String, '/face_data', self.face_callback, 10)
        self.object_sub   = self.create_subscription(String, '/object_data', self.object_callback, 10)
        self.depth_sub    = self.create_subscription(Float32, '/depth_data', self.depth_callback, 10)

        self.behavior_pub = self.create_publisher(String, '/behavior_state', 10) 

        self.current_face   = "looking_forward"
        self.current_object = "none"
        self.current_depth  =  0.5

        self.timer = self.create_timer((0.2), self.analyze_behavior)

    def face_callback(self, msg):
        self.current_face = msg.data

    def object_callback(self, msg):
        self.current_object = msg.data.lower()

    def depth_callback(self,msg):
        self.current_depth = msg.data

    def analyze_behavior(self):
        attention_threshold = self.get_parameter('attention_threshold').get_parameter_value().double_value

        behavior_issues = []

        if self.current_face == "looking away":
            behavior_issues.append("Student is Looking Away !!!")

        if "phone" in self.current_object or "book" in self.current_object:
            behavior_issues.append(f"Prohibited Object Detected: {self.current_object}.")
        
        if self.current_depth > attention_threshold:
            behavior_issues.append("Student is at an unusual distance (too far).")
        elif self.current_depth < 0.2:
            behavior_issues.append("Student is at an unusual distance (too close).")

        state_msg = String()
        
        if len(behavior_issues) > 0:
            # Combine the face + object + depth reasoning 
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
