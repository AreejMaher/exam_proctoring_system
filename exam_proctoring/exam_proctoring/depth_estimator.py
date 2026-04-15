import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from exam_interfaces.msg import DepthData

import cv2
from cv_bridge import CvBridge
import sys
import os
import torch

base_path = os.path.expanduser('~/ros2_ws/src/exam_proctoring/exam_proctoring/')
src_path = os.path.join(base_path,'Depth-Anything-V2/metric_depth/')
sys.path.append(src_path)

from depth_anything_v2.dpt import DepthAnythingV2

class Depth_Estimator(Node):
    def __init__(self):
        super().__init__("depth_estimator")

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
         }

        self.encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        self.depth_model = DepthAnythingV2(**self.model_configs[self.encoder ])
        checkpoint_path = os.path.join(base_path, f'models/depth_anything_v2_{self.encoder}.pth')
        self.depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()
        
        self.depth_threshold = 0.0

        self.create_subscription(Image, "/camera_frames", self.camera_callback, 10)
        self.depth_pub = self.create_publisher(DepthData, "/depth_data", 10)
        self.declare_parameter('depth_threshold', self.depth_threshold)

    def camera_callback(self, img):
        frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        depth_map = self.depth_model.infer_image(frame) # HxW raw depth map in numpy
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = self.bridge.cv2_to_imgmsg(depth_normalized, encoding='mono8', header=img.header)
        # depth_map = self.bridge.cv2_to_compressed_imgmsg(depth_normalized, encoding='mono8', header=img.header)
        depth_msg = DepthData()
        depth_msg.depth_map = depth_map
        depth_msg.distance = 0.0
        self.depth_pub.publish(depth_msg)

        # Visualization 
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imshow('Depth Map', depth_colormap)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Depth_Estimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
