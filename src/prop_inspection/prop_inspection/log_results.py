import rclpy
from rclpy.node import Node


class LogResultsNode(Node):
    def __init__(self):
        super().__init__('log_results')
        self.get_logger().info('log_results node started')


def main(args=None):
    rclpy.init(args=args)
    node = LogResultsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
