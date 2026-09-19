#!/usr/bin/env python3

# MIT License
#
# Vision-enabled chat demo:
#   - subscribe to a camera image topic (default: /image_raw) and keep the
#     latest frame only,
#   - subscribe to a text chat topic (default: /chat_input),
#   - when a chat prompt arrives, attach the latest frame and ask the (v)LLM,
#   - publish the full response on /chat_response and streaming tokens on
#     /chat_response_partial.
#
# Designed for low-CPU machines: the image is resized before being sent, only
# the latest frame is kept, and new prompts are dropped while a previous
# generation is still running.

import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String

from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse


class ChatVisionTopicDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_vision_topic_demo_node")

        # -- Parameters --------------------------------------------------
        self.declare_parameter("input_topic", "chat_input")
        self.declare_parameter("response_topic", "chat_response")
        self.declare_parameter("partial_topic", "chat_response_partial")
        self.declare_parameter("image_topic", "/image_raw")
        # Subscribe to sensor_msgs/CompressedImage instead of raw Image.
        # When true, the default image_topic becomes "/image_raw/compressed"
        # unless the user overrode image_topic explicitly.
        self.declare_parameter("use_compressed", False)
        self.declare_parameter("temp", 0.2)
        self.declare_parameter("reset", True)
        # Resize the longest side of the frame to this many pixels before
        # sending it to the LLM (0 or negative disables resizing).
        self.declare_parameter("image_max_size", 448)
        # Minimum seconds between two generations. Prevents back-to-back
        # heavy loads on weak CPUs.
        self.declare_parameter("cooldown_sec", 0.0)
        # Prompt template. `{user}` is replaced by the /chat_input message.
        # <__media__> is the llama_ros placeholder where the image is
        # injected. If the user prompt already contains <__media__>, the
        # template is bypassed.
        self.declare_parameter(
            "prompt_template",
            "<__media__>{user}",
        )
        # If True and no image has been received yet, still forward the
        # prompt to the LLM as text-only.
        self.declare_parameter("allow_text_only", True)

        input_topic = self.get_parameter("input_topic").value
        response_topic = self.get_parameter("response_topic").value
        partial_topic = self.get_parameter("partial_topic").value
        image_topic = self.get_parameter("image_topic").value
        self.use_compressed = bool(self.get_parameter("use_compressed").value)
        # If compressed mode is requested and the user kept the default raw
        # topic, silently switch to the conventional compressed suffix.
        if self.use_compressed and image_topic == "/image_raw":
            image_topic = "/image_raw/compressed"
        self.temp = float(self.get_parameter("temp").value)
        self.reset = bool(self.get_parameter("reset").value)
        self.image_max_size = int(self.get_parameter("image_max_size").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self.prompt_template = str(self.get_parameter("prompt_template").value)
        self.allow_text_only = bool(self.get_parameter("allow_text_only").value)

        # -- State -------------------------------------------------------
        self._bridge = CvBridge()
        self._image_lock = threading.Lock()
        self._latest_image: Image | None = None
        self._latest_stamp: float = 0.0
        self._busy = False
        self._last_finish_ts: float = 0.0

        # -- llama_ros client (owns its own executor internally) --------
        self.llama_client = LlamaClientNode.get_instance()

        # -- ROS interfaces ---------------------------------------------
        self.response_pub = self.create_publisher(String, response_topic, 10)
        self.partial_pub = self.create_publisher(String, partial_topic, 10)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        if self.use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage, image_topic, self._compressed_cb, image_qos
            )
        else:
            self.image_sub = self.create_subscription(
                Image, image_topic, self._image_cb, image_qos
            )
        self.chat_sub = self.create_subscription(
            String, input_topic, self._chat_cb, 10
        )

        self.get_logger().info(
            f"Vision chat ready. image='{image_topic}', "
            f"input='{input_topic}', response='{response_topic}', "
            f"partial='{partial_topic}', image_max_size={self.image_max_size}"
        )

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image) -> None:
        # Keep the raw ROS Image; resize lazily only when we actually use
        # it. This keeps the image callback O(1) and cheap.
        with self._image_lock:
            self._latest_image = msg
            self._latest_stamp = time.time()

    def _compressed_cb(self, msg: CompressedImage) -> None:
        # Decode jpeg/png -> cv2 -> sensor_msgs/Image so the rest of the
        # pipeline (resize, cv_bridge re-encode) stays unchanged.
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if cv_img is None:
                self.get_logger().warn("CompressedImage decode returned None")
                return
            img = self._bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            img.header = msg.header
        except Exception as e:
            self.get_logger().warn(f"CompressedImage decode failed: {e}")
            return
        with self._image_lock:
            self._latest_image = img
            self._latest_stamp = time.time()

    def _prepare_image_msg(self, img_msg: Image) -> Image | None:
        """Convert to cv2, downscale, convert back to sensor_msgs/Image."""
        try:
            cv_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return None

        if self.image_max_size > 0:
            h, w = cv_img.shape[:2]
            longest = max(h, w)
            if longest > self.image_max_size:
                scale = self.image_max_size / float(longest)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                cv_img = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_AREA)

        try:
            return self._bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge re-encode failed: {e}")
            return None

    # ------------------------------------------------------------------
    # LLM streaming feedback
    # ------------------------------------------------------------------
    def _partial_cb(self, feedback) -> None:
        text = feedback.feedback.partial_response.text
        msg = String()
        msg.data = text
        self.partial_pub.publish(msg)

    # ------------------------------------------------------------------
    # Chat handling
    # ------------------------------------------------------------------
    def _chat_cb(self, msg: String) -> None:
        if self._busy:
            self.get_logger().warn(
                "Still generating a previous response; dropping new prompt."
            )
            return

        now = time.time()
        if now - self._last_finish_ts < self.cooldown_sec:
            self.get_logger().warn("Cooldown in effect; dropping new prompt.")
            return

        user_text = (msg.data or "").strip()
        if not user_text:
            return

        # Snapshot the latest frame (do not hold the lock during inference).
        with self._image_lock:
            img_msg = self._latest_image
            img_age = now - self._latest_stamp if self._latest_image else -1.0

        image_for_llm: Image | None = None
        if img_msg is not None:
            image_for_llm = self._prepare_image_msg(img_msg)

        if image_for_llm is None and not self.allow_text_only:
            self.get_logger().warn(
                "No image available yet; dropping prompt (allow_text_only=false)."
            )
            return

        # Build prompt. If user already inserted <__media__> keep it as-is.
        if "<__media__>" in user_text:
            prompt = user_text
        else:
            prompt = self.prompt_template.format(user=user_text)

        # If we have no image, strip the placeholder so the model does not
        # receive a dangling media tag.
        if image_for_llm is None:
            prompt = prompt.replace("<__media__>", "").strip()

        self._busy = True
        self.get_logger().info(
            f"Prompt: {user_text!r} (image_age={img_age:.2f}s, "
            f"attached={'yes' if image_for_llm is not None else 'no'})"
        )

        response_text = ""
        try:
            goal = GenerateResponse.Goal()
            goal.prompt = prompt
            goal.reset = self.reset
            goal.sampling_config.temp = self.temp
            if image_for_llm is not None:
                goal.images.append(image_for_llm)

            t0 = time.time()
            result, _status = self.llama_client.generate_response(
                goal, self._partial_cb
            )
            dt = time.time() - t0

            if result is not None:
                response_text = result.response.text
            self.get_logger().info(
                f"Done in {dt:.2f}s. Response: {response_text!r}"
            )
        except Exception as e:
            # Never let a bad frame / OOM / cancellation kill the node.
            self.get_logger().error(f"generate_response failed: {e}")
        finally:
            out = String()
            out.data = response_text
            self.response_pub.publish(out)
            self._last_finish_ts = time.time()
            self._busy = False


def main() -> None:
    rclpy.init()
    node = ChatVisionTopicDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
