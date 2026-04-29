from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("prop_inspection")

    world_path = PathJoinSubstitution([pkg_share, "worlds", "inspection_world.sdf"])
    config_path = PathJoinSubstitution([pkg_share, "config", "inspection_params.yaml"])

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ),
        launch_arguments={
            "gz_args": world_path,
        }.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/inspection_camera@sensor_msgs/msg/Image@gz.msgs.Image",
        ],
    )

    spawn_grid = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="prop_inspection",
                executable="spawn_grid",
                output="screen",
                parameters=[config_path],
            )
        ]
    )

    # Start inspector after spawn
    inspect_camera = TimerAction(
        period=45.0,
        actions=[
            Node(
                package="prop_inspection",
                executable="inspect_camera",
                output="screen",
                parameters=[config_path],
            )
        ]
    )

    return LaunchDescription([
        gz_sim_launch,
        bridge,
        spawn_grid,
        inspect_camera,
    ])
