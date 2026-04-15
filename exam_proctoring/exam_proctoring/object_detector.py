#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from exam_interfaces.msg import DetectionList, BoundingBox
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera_frames',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(
            DetectionList,
            '/object_data',
            10)


        self.model = YOLO("yolov8n.pt")  # lightweight model

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        results = self.model(frame)

        detections = DetectionList()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]

                # نفلتر بس الحاجات المهمة
                if label in ["cell phone", "book"]:
                    b = box.xyxy[0]

                    bbox = BoundingBox()
                    bbox.x = int(b[0])
                    bbox.y = int(b[1])
                    bbox.width = int(b[2] - b[0])
                    bbox.height = int(b[3] - b[1])

                    detections.boxes.append(bbox)

        self.publisher.publish(detections)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
