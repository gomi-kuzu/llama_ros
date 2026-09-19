#!/usr/bin/env python3

# MIT License
#
# Simple demo: subscribe to a chat topic, generate an LLM response with
# llama_ros, and publish the response on another topic.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse


class ChatTopicDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_topic_demo_node")

        # Parameters
        self.declare_parameter("input_topic", "chat_input")
        self.declare_parameter("response_topic", "chat_response")
        self.declare_parameter("partial_topic", "chat_response_partial")
        self.declare_parameter("temp", 0.2)
        self.declare_parameter("reset", True)

        input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        response_topic = (
            self.get_parameter("response_topic").get_parameter_value().string_value
        )
        partial_topic = (
            self.get_parameter("partial_topic").get_parameter_value().string_value
        )
        self.temp = self.get_parameter("temp").get_parameter_value().double_value
        self.reset = self.get_parameter("reset").get_parameter_value().bool_value

        # llama_ros client (spins its own executor internally)
        self.llama_client = LlamaClientNode.get_instance()

        # ROS interfaces
        self.response_pub = self.create_publisher(String, response_topic, 10)
        self.partial_pub = self.create_publisher(String, partial_topic, 10)
        self.sub = self.create_subscription(
            String, input_topic, self.chat_cb, 10
        )

        self._busy = False

        self.get_logger().info(
            f"Listening on '{input_topic}', publishing to '{response_topic}' "
            f"(partial: '{partial_topic}')"
        )

    def _partial_cb(self, feedback) -> None:
        text = feedback.feedback.partial_response.text
        msg = String()
        msg.data = text
        self.partial_pub.publish(msg)

    def chat_cb(self, msg: String) -> None:
        if self._busy:
            self.get_logger().warn("Still generating a previous response; skipping.")
            return

        prompt = msg.data
        if not prompt:
            return

        self._busy = True
        self.get_logger().info(f"Prompt: {prompt}")

        try:
            goal = GenerateResponse.Goal()
            goal.prompt = prompt
            goal.reset = self.reset
            goal.sampling_config.temp = float(self.temp)

            result, _status = self.llama_client.generate_response(
                goal, self._partial_cb
            )

            response_text = result.response.text if result is not None else ""
            self.get_logger().info(f"Response: {response_text}")

            out = String()
            out.data = response_text
            self.response_pub.publish(out)
        finally:
            self._busy = False


def main() -> None:
    rclpy.init()
    node = ChatTopicDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
