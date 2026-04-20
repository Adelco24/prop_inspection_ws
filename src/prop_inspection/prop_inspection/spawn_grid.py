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
        self.declare_parameter('class_names', ['good', 'warped', 'incomplete', 'sinkage'])

        self.rows = self.get_parameter('rows').value
        self.cols = self.get_parameter('cols').value
        self.spacing = self.get_parameter('spacing').value
        self.z_height = self.get_parameter('z_height').value
        self.class_names = self.get_parameter('class_names').value

        self.package_share = get_package_share_directory('prop_inspection')
        self.models_dir = os.path.join(self.package_share, 'models')

        self.truth_map = []

        self.spawn_all()

    def spawn_all(self):
        x0 = -((self.cols - 1) * self.spacing) / 2.0
        y0 = -((self.rows - 1) * self.spacing) / 2.0

        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                defect_class = random.choice(self.class_names)
                model_name = f'{defect_class}_prop'
                instance_name = f'{model_name}_{r}_{c}'

                x = x0 + c * self.spacing
                y = y0 + r * self.spacing
                z = self.z_height

                self.spawn_model(instance_name, model_name, x, y, z)
                self.truth_map.append((r, c, defect_class))
                count += 1

        self.get_logger().info(f'Spawned {count} prop instances')
        for item in self.truth_map:
            self.get_logger().info(f'cell=({item[0]},{item[1]}) truth={item[2]}')

    def spawn_model(self, instance_name, model_name, x, y, z):
        model_file = os.path.join(self.models_dir, model_name, 'model.sdf')

        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', 'inspection_world',
            '-name', instance_name,
            '-file', model_file,
            '-x', str(x),
            '-y', str(y),
            '-z', str(z)
        ]

        try:
            subprocess.run(cmd, check=True)
            self.get_logger().info(f'Spawned {instance_name} at ({x:.2f}, {y:.2f}, {z:.2f})')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to spawn {instance_name}: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SpawnGridNode()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
