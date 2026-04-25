import os

import cv2
import numpy as np

import rclpy
import json
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


class InspectCameraNode(Node):
    def __init__(self):
        super().__init__('inspect_camera')

        self.declare_parameter('rows', 5)
        self.declare_parameter('cols', 5)

        self.declare_parameter('cell_w_px', 160)
        self.declare_parameter('cell_h_px', 160)

        self.declare_parameter('grid_center_offset_x_px', 0)
        self.declare_parameter('grid_center_offset_y_px', 0)

        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('save_crops', True)
        self.declare_parameter('crop_margin_px', 10)

        self.rows = int(self.get_parameter('rows').value)
        self.cols = int(self.get_parameter('cols').value)

        self.cell_w_px = int(self.get_parameter('cell_w_px').value)
        self.cell_h_px = int(self.get_parameter('cell_h_px').value)

        self.grid_center_offset_x_px = int(
            self.get_parameter('grid_center_offset_x_px').value
        )
        self.grid_center_offset_y_px = int(
            self.get_parameter('grid_center_offset_y_px').value
        )

        self.save_debug_images = bool(self.get_parameter('save_debug_images').value)
        self.save_crops = bool(self.get_parameter('save_crops').value)
        self.crop_margin_px = int(self.get_parameter('crop_margin_px').value)

        self.bridge = CvBridge()
        self.received_once = False

        pkg_share = get_package_share_directory('prop_inspection')
        self.output_dir = os.path.join(pkg_share, 'debug_output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.template_dir = os.path.join(pkg_share, 'templates')
        self.templates = self.load_templates()

        self.truth_map = None

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.truth_sub = self.create_subscription(
            String,
            '/prop_truth_map',
            self.truth_callback,
            qos
        )
        
        self.subscription = self.create_subscription(
            Image,
            '/inspection_camera',
            self.image_callback,
            10
        )

        self.get_logger().info('inspect_camera node started')
        self.get_logger().info(
            f'Grid config: rows={self.rows}, cols={self.cols}, '
            f'cell={self.cell_w_px}x{self.cell_h_px}px'
        )

    def compute_area_ratio(self, img, contour):
        if img is None or contour is None:
            return 0.0

        h, w = img.shape[:2]
        image_area = float(h * w)

        if image_area <= 0:
            return 0.0

        contour_area = cv2.contourArea(contour)
        return contour_area / image_area
    
    def load_templates(self):
        template_names = ['good', 'warped', 'incomplete', 'sinkage']
        templates = {}

        for name in template_names:
            path = os.path.join(self.template_dir, f'{name}.png')

            if not os.path.exists(path):
                self.get_logger().warn(f'Template missing: {path}')
                continue

            img = cv2.imread(path)

            if img is None:
                self.get_logger().warn(f'Could not read template: {path}')
                continue

            contour = self.extract_main_contour(img)

            if contour is None:
                self.get_logger().warn(f'No contour found in template: {path}')
                continue

            area_ratio = self.compute_area_ratio(img, contour)
            templates[name] = {
                'contour': contour,
                'area_ratio': area_ratio
            }
            self.get_logger().info(f'Loaded template: {name}, area_ratio={area_ratio:.4f}')
            self.get_logger().info(f'Loaded template: {name}')

        self.get_logger().info(f'Loaded {len(templates)} templates')
        return templates


    def extract_main_contour(self, img):
        if img is None or img.size == 0:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Ensure object is white and background is black.
        white_ratio = np.count_nonzero(thresh) / thresh.size
        if white_ratio > 0.5:
            thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 100:
            return None

        return largest
    
    def image_callback(self, msg):
        if self.received_once:
            return
        
        if self.truth_map is None:
            self.get_logger().warn('Image received, but truth map not available yet. Waiting...')
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        h, w, _ = frame.shape
        image_center_x = w // 2
        image_center_y = h // 2

        grid_center_x = image_center_x + self.grid_center_offset_x_px
        grid_center_y = image_center_y + self.grid_center_offset_y_px

        grid_w = self.cols * self.cell_w_px
        grid_h = self.rows * self.cell_h_px

        grid_x_min = int(grid_center_x - grid_w / 2)
        grid_y_min = int(grid_center_y - grid_h / 2)
        grid_x_max = int(grid_center_x + grid_w / 2)
        grid_y_max = int(grid_center_y + grid_h / 2)

        self.get_logger().info(f'Received image: {w}x{h}')
        self.get_logger().info(
            f'Image center: ({image_center_x}, {image_center_y})'
        )
        self.get_logger().info(
            f'Grid center: ({grid_center_x}, {grid_center_y})'
        )
        self.get_logger().info(
            f'Grid bounds: x={grid_x_min}:{grid_x_max}, '
            f'y={grid_y_min}:{grid_y_max}'
        )

        annotated = frame.copy()

        # Draw overall grid boundary.
        cv2.rectangle(
            annotated,
            (grid_x_min, grid_y_min),
            (grid_x_max, grid_y_max),
            (255, 0, 0),
            3
        )

        # Draw image center and grid center.
        cv2.circle(annotated, (image_center_x, image_center_y), 5, (255, 255, 255), -1)
        cv2.circle(annotated, (grid_center_x, grid_center_y), 5, (0, 0, 255), -1)

        predictions = {}
        defective_predictions = []
        correct = 0
        total = 0
        
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = grid_x_min + c * self.cell_w_px
                y1 = grid_y_min + r * self.cell_h_px
                x2 = x1 + self.cell_w_px
                y2 = y1 + self.cell_h_px

                # Clamp crop coordinates to image bounds.
                x1_clamped = max(0, min(w, x1))
                y1_clamped = max(0, min(h, y1))
                x2_clamped = max(0, min(w, x2))
                y2_clamped = max(0, min(h, y2))

                margin = self.crop_margin_px
                cx1 = max(x1_clamped + margin, 0)
                cy1 = max(y1_clamped + margin, 0)
                cx2 = min(x2_clamped - margin, w)
                cy2 = min(y2_clamped - margin, h)

                if cx2 <= cx1 or cy2 <= cy1:
                    label = 'out_of_frame'
                    crop = None
                else:
                    crop = frame[cy1:cy2, cx1:cx2]
                    label = self.classify_crop(crop)

                self.get_logger().info(f'cell=({r},{c}) pred={label}')
                predictions[(r, c)] = label

                if label != 'good':
                    defective_predictions.append((r, c, label))

                if self.truth_map is not None:
                    truth = self.truth_map.get((r, c), None)
                    if truth is not None:
                        total += 1
                        if truth == label:
                            correct += 1

                if self.save_crops and crop is not None:
                    crop_name = f'crop_r{r}_c{c}_{label}.png'
                    cv2.imwrite(os.path.join(self.output_dir, crop_name), crop)

                # Draw unclamped planned cell rectangle so you can see alignment.
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 8, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        self.get_logger().info('PREDICTED GRID FROM CAMERA IMAGE:')
        for r in range(self.rows):
            row_labels = []
            for c in range(self.cols):
                row_labels.append(predictions.get((r, c), 'none'))
            self.get_logger().info(f'  row {r}: {row_labels}')
        self.get_logger().info('Predicted defective cells:')
        if defective_predictions:
            for r, c, pred in defective_predictions:
                if self.truth_map is not None:
                    truth = self.truth_map.get((r, c), 'unknown')
                    self.get_logger().info(
                        f'  cell=({r},{c}) pred={pred}, truth={truth}'
                    )
                else:
                    self.get_logger().info(
                        f'  cell=({r},{c}) pred={pred}'
                    )
        else:
            self.get_logger().info('  none')

        if self.truth_map is not None and total > 0:
            accuracy = 100.0 * correct / total
            self.get_logger().info(
                f'Run accuracy: {correct}/{total} = {accuracy:.1f}%'
            )
        else:
            self.get_logger().warn(
                'No truth map available, so accuracy was not computed'
            )
        
        if self.save_debug_images:
            annotated_path = os.path.join(self.output_dir, 'inspection_annotated.png')
            raw_path = os.path.join(self.output_dir, 'inspection_raw.png')

            cv2.imwrite(annotated_path, annotated)
            cv2.imwrite(raw_path, frame)

            self.get_logger().info(f'Saved {annotated_path}')
            self.get_logger().info(f'Saved {raw_path}')

        self.received_once = True

    def truth_callback(self, msg):
        try:
            truth_list = json.loads(msg.data)
            self.truth_map = {
                (int(item["row"]), int(item["col"])): item["label"]
                for item in truth_list
            }
            self.get_logger().info(f'Received truth map with {len(self.truth_map)} cells')
            self.get_logger().info('TRUTH GRID RECEIVED BY INSPECT_CAMERA:')
            for r in range(self.rows):
                row_labels = []
                for c in range(self.cols):
                    row_labels.append(self.truth_map.get((r, c), 'none'))
                self.get_logger().info(f'  row {r}: {row_labels}')
        except Exception as e:
            self.get_logger().error(f'Failed to parse truth map: {e}')
    
    def classify_crop(self, crop):
        if crop is None or crop.size == 0:
            return 'empty'

        if not self.templates:
            return 'no_templates'

        crop_contour = self.extract_main_contour(crop)

        if crop_contour is None:
            return 'empty'

        crop_area_ratio = self.compute_area_ratio(crop, crop_contour)

        scores = {}

        for label, template_data in self.templates.items():
            template_contour = template_data['contour']
            template_area_ratio = template_data['area_ratio']

            shape_score = cv2.matchShapes(
                crop_contour,
                template_contour,
                cv2.CONTOURS_MATCH_I1,
                0.0
            )

            area_score = abs(crop_area_ratio - template_area_ratio)

            # Combined score:
            # shape_score handles outline
            # area_score breaks ties between similar silhouettes
            combined_score = shape_score + 2.0 * area_score

            scores[label] = combined_score

        best_label = min(scores, key=scores.get)

        debug_scores = ', '.join(
            [f'{label}:{score:.5f}' for label, score in scores.items()]
        )
        self.get_logger().info(
            f'crop_area_ratio={crop_area_ratio:.4f}, scores=({debug_scores}), best={best_label}'
        )

        return best_label
        

def main(args=None):
    rclpy.init(args=args)
    node = InspectCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
