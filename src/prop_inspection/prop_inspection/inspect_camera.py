import os
import cv2
import numpy as np
import time
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
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

        self.declare_parameter('dark_threshold', 80)

        # If best and second-best dark-ratio scores are within this,
        # use probe-template tiebreaker.
        self.declare_parameter('ratio_tie_abs', 0.001)

        self.declare_parameter('good_warped_probe_x', 64)
        self.declare_parameter('good_warped_probe_y', 64)
        self.declare_parameter('sinkage_incomplete_probe_x', 64)
        self.declare_parameter('sinkage_incomplete_probe_y', 64)
        self.declare_parameter('probe_radius_px', 4)

        self.rows = int(self.get_parameter('rows').value)
        self.cols = int(self.get_parameter('cols').value)
        self.cell_w_px = int(self.get_parameter('cell_w_px').value)
        self.cell_h_px = int(self.get_parameter('cell_h_px').value)
        self.grid_center_offset_x_px = int(self.get_parameter('grid_center_offset_x_px').value)
        self.grid_center_offset_y_px = int(self.get_parameter('grid_center_offset_y_px').value)
        self.save_debug_images = bool(self.get_parameter('save_debug_images').value)
        self.save_crops = bool(self.get_parameter('save_crops').value)
        self.crop_margin_px = int(self.get_parameter('crop_margin_px').value)

        self.dark_threshold = int(self.get_parameter('dark_threshold').value)
        self.ratio_tie_abs = float(self.get_parameter('ratio_tie_abs').value)

        self.good_warped_probe_x = int(self.get_parameter('good_warped_probe_x').value)
        self.good_warped_probe_y = int(self.get_parameter('good_warped_probe_y').value)
        self.sinkage_incomplete_probe_x = int(self.get_parameter('sinkage_incomplete_probe_x').value)
        self.sinkage_incomplete_probe_y = int(self.get_parameter('sinkage_incomplete_probe_y').value)
        self.probe_radius_px = int(self.get_parameter('probe_radius_px').value)

        self.bridge = CvBridge()
        self.received_once = False
        self.truth_map = None

        pkg_share = get_package_share_directory('prop_inspection')
        self.output_dir = os.path.join(pkg_share, 'debug_output')
        os.makedirs(self.output_dir, exist_ok=True)

        self.template_dir = os.path.join(pkg_share, 'templates')
        self.templates = self.load_templates()

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

            dark_ratio = self.compute_dark_ratio(img)

            templates[name] = {
                'image': img,
                'dark_ratio': dark_ratio
            }

            self.get_logger().info(
                f'Loaded template: {name}, dark_ratio={dark_ratio:.5f}'
            )

        self.get_logger().info(f'Loaded {len(templates)} templates')
        return templates

    def image_callback(self, msg):
        if self.received_once:
            return

        if self.truth_map is None:
            self.get_logger().warn('Image received, but truth map not available yet. Waiting...')
            return

        start_time = time.perf_counter()

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

        annotated = frame.copy()

        cv2.rectangle(
            annotated,
            (grid_x_min, grid_y_min),
            (grid_x_max, grid_y_max),
            (255, 0, 0),
            3
        )

        cv2.circle(annotated, (image_center_x, image_center_y), 2, (255, 255, 255), -1)
        cv2.circle(annotated, (grid_center_x, grid_center_y), 2, (0, 0, 255), -1)

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
                    label = self.classify_crop(crop, r, c)

                predictions[(r, c)] = label

                if label != 'good':
                    defective_predictions.append((r, c, label))

                truth = self.truth_map.get((r, c), None)
                if truth is not None:
                    total += 1

                    truth_is_bad = truth != 'good'
                    pred_is_bad = label != 'good'

                    if truth_is_bad == pred_is_bad:
                        correct += 1

                if self.save_crops and crop is not None:
                    crop_name = f'crop_r{r}_c{c}_{label}.png'
                    cv2.imwrite(os.path.join(self.output_dir, crop_name), crop)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Probe markers: magenta = good/warped, yellow = sinkage/incomplete.
                cv2.circle(
                    annotated,
                    (
                        x1 + self.crop_margin_px + self.good_warped_probe_x,
                        y1 + self.crop_margin_px + self.good_warped_probe_y
                    ),
                    2,
                    (255, 0, 255),
                    -1
                )

                cv2.circle(
                    annotated,
                    (
                        x1 + self.crop_margin_px + self.sinkage_incomplete_probe_x,
                        y1 + self.crop_margin_px + self.sinkage_incomplete_probe_y
                    ),
                    2,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 8, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        elapsed = time.perf_counter() - start_time

        self.get_logger().info('PREDICTED GRID FROM CAMERA IMAGE:')
        for r in range(self.rows):
            row_labels = [predictions.get((r, c), 'none') for c in range(self.cols)]
            self.get_logger().info(f'  row {r}: {row_labels}')

        self.get_logger().info('Predicted defective cells:')
        if defective_predictions:
            for r, c, pred in defective_predictions:
                truth = self.truth_map.get((r, c), 'unknown')
                self.get_logger().info(f'  cell=({r},{c}) pred={pred}, truth={truth}')
        else:
            self.get_logger().info('  none')

        if total > 0:
            accuracy = 100.0 * correct / total
            self.get_logger().info(
                f'Good/bad accuracy: {correct}/{total} = {accuracy:.1f}% in {elapsed:.4f} seconds.'
            )
        else:
            self.get_logger().warn('No truth cells available, so accuracy was not computed')

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
                row_labels = [self.truth_map.get((r, c), 'none') for c in range(self.cols)]
                self.get_logger().info(f'  row {r}: {row_labels}')

        except Exception as e:
            self.get_logger().error(f'Failed to parse truth map: {e}')

    def compute_dark_ratio(self, img):
        if img is None or img.size == 0:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_mask = gray < self.dark_threshold
        return np.count_nonzero(dark_mask) / float(gray.size)

    def classify_crop(self, crop, row, col):
        if crop is None or crop.size == 0:
            return 'empty'

        if not self.templates:
            return 'no_templates'

        crop_dark_ratio = self.compute_dark_ratio(crop)

        scores = {}
        for label, template_data in self.templates.items():
            template_ratio = template_data['dark_ratio']
            scores[label] = abs(crop_dark_ratio - template_ratio)

        sorted_labels = sorted(scores, key=scores.get)
        best = sorted_labels[0]
        second = sorted_labels[1]

        best_score = scores[best]
        second_score = scores[second]

        if abs(second_score - best_score) <= self.ratio_tie_abs:
            label = self.tiebreak_with_probe(crop, best, second)
            method = f'probe_tiebreak({best},{second})'
        else:
            label = best
            method = 'dark_ratio'

        score_text = ', '.join(
            [f'{label_name}:{scores[label_name]:.5f}' for label_name in sorted_labels]
        )

        self.get_logger().info(
            f'cell=({row},{col}) crop_dark_ratio={crop_dark_ratio:.5f}, '
            f'scores=({score_text}), best={best}, second={second}'
        )

        return label

    def tiebreak_with_probe(self, crop, label_a, label_b):
        if self.is_high_group_pair(label_a, label_b):
            x = self.good_warped_probe_x
            y = self.good_warped_probe_y
            pair_name = 'good/warped'
        elif self.is_low_group_pair(label_a, label_b):
            x = self.sinkage_incomplete_probe_x
            y = self.sinkage_incomplete_probe_y
            pair_name = 'sinkage/incomplete'
        else:
            # If the ambiguous pair is not one of the expected pairs,
            # still use the closest-probe rule with the good/warped probe.
            x = self.good_warped_probe_x
            y = self.good_warped_probe_y
            pair_name = f'{label_a}/{label_b}'

        crop_probe = self.get_probe_mean(crop, x, y, self.probe_radius_px)
        a_probe = self.get_template_probe_mean(label_a, x, y, self.probe_radius_px)
        b_probe = self.get_template_probe_mean(label_b, x, y, self.probe_radius_px)

        if a_probe is None or b_probe is None:
            return label_a

        a_error = abs(crop_probe - a_probe)
        b_error = abs(crop_probe - b_probe)

        self.get_logger().info(
            f'  tiebreak pair={pair_name}, crop_probe={crop_probe:.2f}, '
            f'{label_a}_template={a_probe:.2f}, {label_b}_template={b_probe:.2f}, '
            f'{label_a}_err={a_error:.2f}, {label_b}_err={b_error:.2f}'
        )

        if a_error <= b_error:
            return label_a
        return label_b

    def is_high_group_pair(self, label_a, label_b):
        return set([label_a, label_b]) == set(['good', 'warped'])

    def is_low_group_pair(self, label_a, label_b):
        return set([label_a, label_b]) == set(['sinkage', 'incomplete'])

    def get_template_probe_mean(self, template_label, x, y, radius):
        if template_label not in self.templates:
            self.get_logger().warn(f'Missing template for {template_label}')
            return None

        template_img = self.templates[template_label]['image']
        return self.get_probe_mean(template_img, x, y, radius)

    def get_probe_mean(self, crop, x, y, radius):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))

        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        region = gray[y1:y2, x1:x2]
        return float(np.mean(region))


def main(args=None):
    rclpy.init(args=args)
    node = InspectCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
