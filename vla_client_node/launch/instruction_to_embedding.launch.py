from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="vla_client_node",
            executable="instruction_to_embedding",
            name="instruction_to_embedding_node",
            output="screen",
            parameters=[{
                "instruction_topic": "/vla/instruction",
                "embedding_topic": "/vla/goal_embedding",
                "embedding_dim": 128
            }]
        )
    ])