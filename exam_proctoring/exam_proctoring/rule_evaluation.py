#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from exam_interfaces.srv import CheckViolation
import json


class RuleEvaluation(Node):
    def __init__(self):
        super().__init__('rule_evaluation')

        # subscribers
        self.event_sub = self.create_subscription(String, '/behavior_state', self.behavior_callback, 10)

        # publishers
        self.alert_pub = self.create_publisher(String, '/violation_event', 10)

        # service server
        self.srv = self.create_service(CheckViolation, '/check_violation', self.check_violation_callback)

        # parameters
        self.declare_parameter("violation_rules", "strict")
        self.violation_rules = self.get_parameter('violation_rules').value

        self.get_logger().info("rule evaluation node is running.")

        self.latest_behavior = {"status": "Normal", "reasons": []}
        self.violation_active = False

    def behavior_callback(self, msg):
        try:
            self.latest_behavior = json.loads(msg.data)
            
            if self.latest_behavior.get("status") == "Suspicious":
                self.violation_active = True
                
                # Get the reasons list
                reasons = self.latest_behavior.get("reasons", [])
                
                # Convert the list into a clean string (e.g., "Reason 1 | Reason 2")
                # This ensures the Alert Node receives a simple string it can process easily.
                clean_reasons_str = " | ".join(reasons) if reasons else "Suspicious behavior detected"

                v_msg = String()
                v_msg.data = clean_reasons_str
                
                # Now publishing a plain String to /violation_event
                self.alert_pub.publish(v_msg)
                
                self.get_logger().warn(f"Published Violation String: {clean_reasons_str}")

            else:
                self.violation_active = False

        except json.JSONDecodeError:
            self.get_logger().error("Error parsing behavior data.")


    def check_violation_callback(self, request, response):
        reasons = self.latest_behavior.get("reasons", [])
   
        if self.violation_active:
            if self.violation_rules == 'strict':
                if any("Violation:" in r for r in reasons):
                    response.message = "WARNING: verification failed, make sure you are alone in the camera frame and facing forward."

                elif any("Prohibited" in r for r in reasons):
                    response.message = "Cheating Detected: prohibited object found."
                
                elif any("too far" in r for r in reasons):
                    response.message = "WARNING: maintain an appropriate distance from the camera."
                
                elif any("too close" in r for r in reasons):
                    response.message = "WARNING: maintain an appropriate distance from the camera."

            else:
                if any("Violation:" in r for r in reasons):
                    response.message = "make sure you are alone in the camera frame."

                elif any("Prohibited" in r for r in reasons):
                    response.message = "phones and books are prohibited in this exam."
                
                elif any("too far" in r for r in reasons):
                    response.message = "get closer so the system can monitor you clearly."
                
                elif any("too close" in r for r in reasons):
                    response.message = "please get back you are obstructing visibility."
                    
            response.violation_detected = True


        else:
            response.violation_detected = False
            response.message = "Student is focused"

        self.get_logger().info(f"{response.message}")

        return response
    
def main(args=None):
    rclpy.init(args=args)
    node = RuleEvaluation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()