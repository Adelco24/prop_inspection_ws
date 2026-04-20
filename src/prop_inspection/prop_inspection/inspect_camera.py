import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class InspectCameraNode(Node):
    def __init__(self):
        super().__init__('inspect_camera')

        self.declare_parameter('rows', 5)
        self.declare_parameter('cols', 5)
        self.declare_parameter('save_debug_images', False)

        self.rows = self.get_parameter('rows').value
        self.cols = self.get_parameter('cols').value
        self.save_debug_images = self.get_parameter('save_debug_images').value

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/inspection_camera',
            self.image_callback,
            10
        )

        self.received_once = False
        self.get_logger().info('inspect_camera node started')

    def image_callback(self, msg):
        if self.received_once:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        h, w, _ = frame.shape
        cell_h = h // self.rows
        cell_w = w // self.cols

        self.get_logger().info(f'Image size: {w}x{h}')
        self.get_logger().info(f'Cell size: {cell_w}x{cell_h}')

        annotated = frame.copy()

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = (c + 1) * cell_w
                y2 = (r + 1) * cell_h

                crop = frame[y1:y2, x1:x2]
                label = self.simple_classifier(crop)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                self.get_logger().info(f'cell=({r},{c}) pred={label}')

        if self.save_debug_images:
            cv2.imwrite('inspection_debug.png', annotated)
            self.get_logger().info('Saved inspection_debug.png')

        self.received_once = True

    def simple_classifier(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)

        if mean_val > 170:
            return 'good?'
        elif mean_val > 120:
            return 'warped?'
        elif mean_val > 80:
            return 'incomplete?'
        else:
            return 'sinkage?'


def main(args=None):
    rclpy.init(args=args)
    node = InspectCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
