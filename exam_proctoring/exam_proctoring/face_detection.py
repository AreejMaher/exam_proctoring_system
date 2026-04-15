import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from exam_interfaces.msg import FaceData

class Face_Detection(Node):
    def __init__(self):
        super().__init__("face_detection")
        self.get_logger().info(f"Face Detection Node has started.")

        self.bridge = CvBridge()
        face_xml= cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.declare_parameter('scale_factor', 1.1)
        self.declare_parameter('min_neighbors', 9)
        self.scaleFactor = self.get_parameter('scale_factor').value
        self.minNeighbors = self.get_parameter('min_neighbors').value
        if self.face_cascade.empty():
            self.get_logger().info("Coulding load the HAAR XML file!!!!")

        self.create_subscription(Image, "/camera_frames", self.camera_callback, 10)
        self.face_pub = self.create_publisher(FaceData, "/face_data", 10)

    def camera_callback(self, img):
        try:
            frame = self.bridge.imgmsg_to_cv2(img, desired_encoding= "bgr8")
            # self.get_logger().info(f"frame (height, width, channel) :  {frame.shape}")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray_frame, self.scaleFactor, self.minNeighbors)
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            face_msg = FaceData()

            if (len(faces) > 0):
                face_msg.face_detected = True
                (fx,fy,fw,fh) = faces[0] # as only one student per exam
            else:
                face_msg.face_detected = False
                fx = fy = fw = fh = 0
            face_msg.face_count = len(faces) # for violation (when zero or more than one)
            face_msg.x = int(fx)
            face_msg.y = int(fy)
            face_msg.w = int(fw)
            face_msg.h = int(fh)
            self.face_pub.publish(face_msg)


            cv2.imshow('img',frame)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            
        except:
            self.get_logger().error(f"Conversion failed!!!!!!")

def main(args=None):
    rclpy.init(args=args)
    node = Face_Detection()
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