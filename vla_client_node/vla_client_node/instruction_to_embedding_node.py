import hashlib
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray


class InstructionToEmbeddingNode(Node):
    def __init__(self):
        super().__init__("instruction_to_embedding_node")

        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("embedding_topic", "/vla/goal_embedding")
        self.declare_parameter("embedding_dim", 4096)

        self.embedding_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("embedding_topic").value,
            10,
        )

        self.create_subscription(
            String,
            self.get_parameter("instruction_topic").value,
            self.instruction_callback,
            10,
        )

        self.get_logger().info("instruction_to_embedding_node started")
        self.get_logger().info("Publish an instruction on /vla/instruction to generate a 4096-d embedding")

    def make_embedding(self, text: str, dim: int) -> np.ndarray:
        """
        Deterministic embedding for pipeline testing only.
        Replace this with UniVLA server output later.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little", signed=False)
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(dim).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def instruction_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        dim = int(self.get_parameter("embedding_dim").value)
        emb = self.make_embedding(text, dim)

        out = Float32MultiArray()
        out.data = emb.tolist()
        self.embedding_pub.publish(out)

        self.get_logger().info(f"Instruction received: {text}")
        self.get_logger().info(f"Published latent embedding dim: {dim}")


def main():
    rclpy.init()
    node = InstructionToEmbeddingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()