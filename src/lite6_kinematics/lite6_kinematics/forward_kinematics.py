"""Labwork 1 - forward kinematics of the UFACTORY Lite 6, as a ROS 2 node.

THIS IS THE ONLY FILE YOU EDIT IN LABWORK 1.

Implement the following, replacing every ``raise NotImplementedError``:

    Part B - the maths (no ROS needed)
        dh_table(q)                        the modified DH table for a joint vector
        mdh_transform(a, alpha, d, theta)  one link transform, ^{i-1}_i T
        forward_kinematics(q, all_frames)  ^0_6 T, or all seven frames ^0_i T

    Part C - the node
        FkNode                             reads /joint_states, publishes the
                                           pose of the end-effector

Run the node with ``ros2 run lite6_kinematics fk_node``.

``pose_from_transform``, ``quaternion_from_matrix`` and ``main`` at the bottom
are PROVIDED: already implemented, do not edit them.

Keep the function and class names, arguments and return types exactly as they
are: the checker node (``fk_check``), the tests in ``test/`` and the
instructor's grader all use them by name.

Conventions - fixed, not yours to choose:

    * Craig's MODIFIED Denavit-Hartenberg convention, frames {0} ... {6}.
    * Frame {0} coincides with the ROS frame ``link_base``.
    * Frame {6} coincides with the ROS frame ``link_eef`` (the end-effector
      frame).
    * Joint angles in RADIANS, exactly as published on /joint_states
      (q = 0 is the controller's zero configuration).
    * Lengths in METRES.

Allowed imports: ``numpy`` and ``math`` for the maths, and ``rclpy``,
``sensor_msgs`` and ``geometry_msgs`` for the node (already imported below).
Nothing else - no MoveIt, no kinematics library, no URDF parser, no tf.
``numpy`` does the linear algebra (``@`` for matrix products, ``np.cos``,
``np.eye``...); the robotics is yours.
``test/test_forward_kinematics.py`` checks the imports.
"""

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError:         # the maths and its unit tests also run without ROS
    rclpy = None
    Node = object

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
BASE_FRAME = 'link_base'


# =============================================================================
# Part B - the maths
# =============================================================================
# =============================================================================
# Part B - the maths
# =============================================================================
import math

def dh_table(q) -> np.ndarray:
    """Modified DH parameters of the Lite 6 for joint vector q."""
    return np.array([
        [0.0,       0.0,           0.2433,  q[0] + 0.0],
        [0.0,       -math.pi/2,    0.0,     q[1] - math.pi/2],
        [0.200,     math.pi,       0.0,     q[2] - math.pi/2],
        [0.087,     math.pi/2,     0.2276,  q[3] + 0.0],
        [0.0,       math.pi/2,     0.0,     q[4] + 0.0],
        [0.0,       -math.pi/2,    0.0615,  q[5] + 0.0]
    ])

def mdh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Transform from frame {i-1} to frame {i}, ^{i-1}_i T, modified DH."""
    c_theta = math.cos(theta)
    s_theta = math.sin(theta)
    c_alpha = math.cos(alpha)
    s_alpha = math.sin(alpha)
    
    return np.array([
        [c_theta,            -s_theta,             0,             a],
        [s_theta * c_alpha,  c_theta * c_alpha,   -s_alpha,      -d * s_alpha],
        [s_theta * s_alpha,  c_theta * s_alpha,    c_alpha,       d * c_alpha],
        [0.0,                0.0,                  0.0,           1.0]
    ])


def forward_kinematics(q, all_frames: bool = False):
    """Pose of the end-effector, ^0_6 T, for joint vector q."""
    T = np.eye(4)
    frames = [T]
    
    table = dh_table(q)
    for row in table:
        T_i = mdh_transform(row[0], row[1], row[2], row[3])
        T = T @ T_i
        frames.append(T)
        
    if all_frames:
        return frames
    return T

# =============================================================================
# Part C - the node
# =============================================================================
# =============================================================================
# Part C - the node
# =============================================================================
class FkNode(Node):
    def __init__(self, joint_states_topic='/joint_states', pose_topic='/lite6_fk/pose'):
        super().__init__('fk_node')
        
        # 1. Criar o subscritor e o publicador usando os argumentos fornecidos
        self.subscription = self.create_subscription(
            JointState,
            joint_states_topic,
            self.on_joint_state,
            10
        )
        self.publisher = self.create_publisher(
            PoseStamped,
            pose_topic,
            10
        )

    def on_joint_state(self, msg):
        """Called for every JointState: compute the pose and publish it."""
        try:
            # 1. Extrai os ângulos POR NOME usando a lista JOINT_NAMES
            q = [msg.position[msg.name.index(name)] for name in JOINT_NAMES]
        except ValueError:
            # Ignora se faltar alguma das 6 juntas na mensagem
            return

        # 2. T = forward_kinematics(q)
        T = forward_kinematics(q)

        # 3. Preenche a mensagem PoseStamped
        stamped_msg = PoseStamped()
        stamped_msg.header.frame_id = BASE_FRAME
        stamped_msg.header.stamp = msg.header.stamp  # Copia o carimbo temporal

        # Preenche a posição a partir da última coluna da matriz T
        stamped_msg.pose.position.x = float(T[0, 3])
        stamped_msg.pose.position.y = float(T[1, 3])
        stamped_msg.pose.position.z = float(T[2, 3])

        # Usa quaternion_from_matrix para converter a rotação 3x3 num quaternion
        q_rot = quaternion_from_matrix(T[:3, :3])
        stamped_msg.pose.orientation.x = float(q_rot[0])
        stamped_msg.pose.orientation.y = float(q_rot[1])
        stamped_msg.pose.orientation.z = float(q_rot[2])
        stamped_msg.pose.orientation.w = float(q_rot[3])

        # 4. Publica
        self.publisher.publish(stamped_msg)


# =============================================================================
# PROVIDED - DO NOT EDIT. Already implemented; nothing below is yours to change.
# =============================================================================
def pose_from_transform(T) -> np.ndarray:
    """End-effector pose as six numbers.

    Converts ^0_6 T into the representation used by the Lite 6 controller and
    by UFACTORY Studio. fk_check uses it to print your pose.

    Args:
        T: (4, 4) homogeneous transform.

    Returns:
        (6,) array [x, y, z, roll, pitch, yaw]: position in metres, orientation
        in radians as X-Y-Z FIXED angles (Craig, section 2.8):

            R = Rz(yaw) @ Ry(pitch) @ Rx(roll),   -pi/2 <= pitch <= pi/2

        (UFACTORY Studio shows the same three angles, in degrees.) At
        pitch = +-pi/2 exactly, roll and yaw cannot be told apart; this
        function takes the degenerate-case choice of Craig 2.8: yaw = 0.
    """
    T = np.asarray(T, dtype=float)
    r = T[:3, :3]
    pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
    cb = np.cos(pitch)
    if cb > 1e-12:
        yaw = np.arctan2(r[1, 0] / cb, r[0, 0] / cb)
        roll = np.arctan2(r[2, 1] / cb, r[2, 2] / cb)
    elif pitch > 0:            # pitch = +90 deg
        yaw, roll = 0.0, np.arctan2(r[0, 1], r[1, 1])
    else:                      # pitch = -90 deg
        yaw, roll = 0.0, -np.arctan2(r[0, 1], r[1, 1])
    return np.array([T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw])


def quaternion_from_matrix(R) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix, with w >= 0.

    This is the order of geometry_msgs/Quaternion: msg.x, msg.y, msg.z, msg.w.
    """
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        w, x, y, z = s / 4.0, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, s / 4.0, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4.0, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4.0
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def main(args=None):
    """Entry point of ``ros2 run lite6_kinematics fk_node``."""
    if rclpy is None:
        raise SystemExit('ROS 2 is not available here: run this inside the container.')
    rclpy.init(args=args)
    node = None
    status = 0
    try:
        node = FkNode()
        node.get_logger().info('fk_node running: joint states in, pose out')
        rclpy.spin(node)
    except NotImplementedError as e:
        print(f'\n{e}\nImplement it in lite6_kinematics/forward_kinematics.py first.')
        status = 1
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == '__main__':
    raise SystemExit(main())