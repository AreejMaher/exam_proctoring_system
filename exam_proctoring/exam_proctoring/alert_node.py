import rclpy  # ros2 Python client library (ros1: rospy)
from rclpy.node import Node  # base class for ros2 nodes (ros1: no node class)

from rclpy.action import ActionServer  # creates action server in ros2 (ros1: actionlib)
from std_msgs.msg import String  # standard string message type (same in ros1)

from exam_interfaces.action import AlertAction  
# custom action interface
# ros1: imported from package msg folder

import time  # used to simulate alert execution


class AlertNode(Node):  # define ros2 node class
    def __init__(self):
        super().__init__("alert_node")
        # initialize node with name "alert_node"
        # ros1: rospy.init_node("alert_node")

        # Publisher
        self.publisher = self.create_publisher(String, "alert_status", 10)
        # publish alert status
        # ros1: rospy.Publisher("alert_status", String, queue_size=10)

        # Subscriber
        self.create_subscription(String,"violation_event",self.violation_callback,10)
        # subscribe to violation topic
        # ros1: rospy.Subscriber("violation_event", String, callback)

        # Action Server
        self._action_server = ActionServer(self,AlertAction,"alert_action",self.execute_callback)
        # create action server
        # ros1: actionlib.SimpleActionServer()

        # Parameter
        self.declare_parameter("alert_level", "WARNING")
        # declare alert level parameter
        # ros1: rospy.get_param("~alert_level", "WARNING")

        self.alert_level = self.get_parameter("alert_level").value
        # get parameter value

        self.last_violation = ""
        # stores last violation message

        self.get_logger().info("Alert Node Started")
        # ros1: rospy.loginfo(...)

    def violation_callback(self, msg):
        # called when violation message is received

        self.last_violation = msg.data
        # save received message

        self.get_logger().warn(f"Violation received: {msg.data}")
        # ros1: rospy.logwarn(...)
        alert_msg = String()
        alert_msg.data = f"{self.alert_level}: {msg.data}"
    
    def execute_callback(self, goal_handle):
        # called when action goal is sent

        message = goal_handle.request.message
        # get message from action goal
        # ros1: goal.message

        feedback_msg = AlertAction.Feedback()
        # create feedback object

        for i in range(5):
            feedback_msg.feedback = f"Alert running {i}"
            # update feedback text

            goal_handle.publish_feedback(feedback_msg)
            # send feedback to client
            # ros1: publish_feedback()

            time.sleep(1)
            # simulate action delay

        goal_handle.succeed()
        # mark goal as completed
        # ros1: set_succeeded()

        alert_msg = String()
        alert_msg.data = f"{self.alert_level}: {message}"
        # create published alert message

        self.publisher.publish(alert_msg)
        # publish alert status
        # ros1: same publish()

        result = AlertAction.Result()
        # create result object

        result.result = "Alert completed"
        # final action result

        return result
        # return result to client

def main(args=None):
    rclpy.init(args=args)
    # initialize ros2
    # ros1: rospy.init_node()

    node = AlertNode()
    # create node object

    try:
        rclpy.spin(node)
        # keep node running
        # ros1: rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # destroy node
        # ros2 only

        rclpy.shutdown()
        # shutdown ros2
        # ros1 handles automatically


if __name__ == "__main__":
    main()


# command to run node
# ros2 run my_py_pkg alert_node

# publish test violation
# ros2 topic pub /violation_event std_msgs/String "data: 'Phone detected'"

# send action goal
# ros2 action send_goal /alert_action my_robot_interfaces/action/AlertAction "{message: 'Phone detected'}"