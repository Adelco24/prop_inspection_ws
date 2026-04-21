import os
import random
import subprocess

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class SpawnGridNode(Node):
    def __init__(self):
        super().__init__('spawn_grid')

        self.declare_parameter('rows', 5)
        self.declare_parameter('cols', 5)
        self.declare_parameter('spacing', 0.22)
        self.declare_parameter('z_height', 0.05)
        self.declare_parameter('spawn_period', 0.75)
        self.declare_parameter('good_fraction', 0.75)
        self.declare_parameter('class_names', ['good', 'warped', 'incomplete', 'sinkage'])

        self.rows = self.get_parameter('rows').value
        self.cols = self.get_parameter('cols').value
        self.spacing = self.get_parameter('spacing').value
        self.z_height = self.get_parameter('z_height').value
        self.spawn_period = self.get_parameter('spawn_period').value
        self.good_fraction = float(self.get_parameter('good_fraction').value)
        self.class_names = self.get_parameter('class_names').value

        self.package_share = get_package_share_directory('prop_inspection')
        self.models_dir = os.path.join(self.package_share, 'models')

        self.truth_map = []
        self.spawn_queue = []
        self.spawn_index = 0

        self.validate_parameters()
        self.build_spawn_queue()

        self.get_logger().info(
            f'Starting timed spawn of {len(self.spawn_queue)} props '
            f'with period {self.spawn_period:.2f} sec'
        )
        self.get_logger().info(
            f'Configured class distribution: good={self.good_fraction:.2f}, '
            f'other defects share remaining {(1.0 - self.good_fraction):.2f}'
        )

        self.timer = self.create_timer(self.spawn_period, self.spawn_next_model)

    def validate_parameters(self):
        if len(self.class_names) != 4:
            raise ValueError(
                "class_names must contain exactly 4 entries: "
                "['good', 'warped', 'incomplete', 'sinkage']"
            )

        if not (0.0 <= self.good_fraction <= 1.0):
            raise ValueError('good_fraction must be between 0.0 and 1.0')

    def choose_defect_class(self):
        good_label = 'good'
        bad_labels = [label for label in self.class_names if label != good_label]

        if len(bad_labels) != 3:
            raise ValueError("Expected exactly 3 bad classes besides 'good'")

        remaining = 1.0 - self.good_fraction
        bad_fraction_each = remaining / len(bad_labels)

        labels = [good_label] + bad_labels
        weights = [self.good_fraction] + [bad_fraction_each] * len(bad_labels)

        return random.choices(labels, weights=weights, k=1)[0]

    def build_spawn_queue(self):
        x0 = -((self.cols - 1) * self.spacing) / 2.0
        y0 = -((self.rows - 1) * self.spacing) / 2.0

        planned_counts = {
            'good': 0,
            'warped': 0,
            'incomplete': 0,
            'sinkage': 0
        }

        for r in range(self.rows):
            for c in range(self.cols):
                defect_class = self.choose_defect_class()
                model_name = f'{defect_class}_prop'
                instance_name = f'{model_name}_{r}_{c}'

                x = x0 + c * self.spacing
                y = y0 + r * self.spacing
                z = self.z_height

                self.spawn_queue.append({
                    'row': r,
                    'col': c,
                    'defect_class': defect_class,
                    'model_name': model_name,
                    'instance_name': instance_name,
                    'x': x,
                    'y': y,
                    'z': z,
                })

                if defect_class in planned_counts:
                    planned_counts[defect_class] += 1

        self.get_logger().info(f'Planned counts: {planned_counts}')

    def spawn_next_model(self):
        if self.spawn_index >= len(self.spawn_queue):
            self.get_logger().info('Finished spawning all props')
            self.timer.cancel()

            for item in self.truth_map:
                self.get_logger().info(
                    f'cell=({item["row"]},{item["col"]}) truth={item["label"]}'
                )
            return

        item = self.spawn_queue[self.spawn_index]

        success = self.spawn_model(
            item['instance_name'],
            item['model_name'],
            item['x'],
            item['y'],
            item['z']
        )

        if success:
            self.truth_map.append({
                'row': item['row'],
                'col': item['col'],
                'label': item['defect_class']
            })

        self.spawn_index += 1

    def spawn_model(self, instance_name, model_name, x, y, z):
        model_file = os.path.join(self.models_dir, model_name, 'model.sdf')

        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', 'inspection_world',
            '-file', model_file,
            '-name', instance_name,
            '-x', str(x),
            '-y', str(y),
            '-z', str(z)
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.get_logger().info(
                f'Spawned {instance_name} at ({x:.2f}, {y:.2f}, {z:.2f})'
            )
            if result.stdout.strip():
                self.get_logger().debug(result.stdout.strip())
            return True

        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to spawn {instance_name}')
            if e.stdout:
                self.get_logger().error(f'stdout: {e.stdout}')
            if e.stderr:
                self.get_logger().error(f'stderr: {e.stderr}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = SpawnGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
