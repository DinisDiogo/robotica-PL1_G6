"""Labwork 2 - algebraic inverse kinematics of the UFACTORY Lite 6, as a ROS 2 node.

THIS IS THE ONLY FILE YOU EDIT IN LABWORK 2.

Implement the following, replacing every ``raise NotImplementedError``:

    Part B - the maths (no ROS needed)
        inverse_kinematics(T_d)                    every distinct solution for a pose
        select_solution(solutions, q_current)      the solution that is executed

    Part B - the node
        IkNode                                     reads a desired pose, publishes
                                                   the selected joint angles

Run the node with ``ros2 run lite6_ik ik_node``.

``transform_from_pose``, ``JOINT_LIMITS``, the constants ``TOL_SAME`` and
``TOL_WRIST`` and ``main`` below are PROVIDED: already implemented, do not edit
them.

Keep the function and class names, arguments and return types exactly as they
are: the checker node (``ik_check``), the study program (``ik_study``), the
tests in ``test/`` and the instructor's grader all use them by name.

Conventions - those of Labwork 1:

    * Frame {0} coincides with the ROS frame ``link_base``; frame {6} with
      ``link_eef``. The desired pose T_d is the pose of {6} in {0}.
    * Joint angles in RADIANS, in the controller's convention, as published on
      /joint_states (q = 0 is the controller's zero configuration).
    * Lengths in METRES.

Allowed imports: ``numpy`` and ``math`` for the maths, ``forward_kinematics``
from your own ``lite6_kinematics.forward_kinematics`` (already imported below),
and ``rclpy``, ``sensor_msgs`` and ``geometry_msgs`` for the node (already
imported below). Nothing else - no MoveIt, no kinematics library, no URDF
parser, no tf. ``test/test_inverse_kinematics.py`` checks the imports.
"""

import math

import numpy as np

from lite6_kinematics.forward_kinematics import dh_table, forward_kinematics, mdh_transform

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
def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval (-pi, pi]."""
    w = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if w <= -math.pi else w


def _solve_trig(a: float, b: float, c: float):
    """Solve a * cos(x) + b * sin(x) = c for x in (-pi, pi]."""
    r = math.hypot(a, b)
    if r < 1e-12:
        return []
    ratio = c / r
    if abs(ratio) > 1.0 + 1e-7:
        return []
    ratio = max(-1.0, min(1.0, ratio))
    phi = math.atan2(b, a)
    delta = math.acos(ratio)
    if delta < 1e-12 or abs(delta - math.pi) < 1e-12:
        return [phi + delta]
    return [phi + delta, phi - delta]


def inverse_kinematics(T_d) -> list:
    """Every distinct joint vector that puts the end-effector at T_d.

    The solution is closed-form, from the expressions of Part A: no iterative
    or optimisation method.

    Args:
        T_d: (4, 4) desired pose of {6} in {0}; translation in metres.

    Returns:
        A list of (6,) float arrays, joint angles in radians, order
        joint1..joint6, one array per distinct solution (in any order). The
        empty list if T_d is out of the robot's reach.

    Conventions of Part A:
        * every angle is in (-pi, pi]; the joint ranges JOINT_LIMITS are NOT
          applied here: a solution outside them is still returned;
        * two solutions in which no angle differs by more than TOL_SAME
          (angles compared modulo 2 pi) are the same solution: only one of
          them is returned;
        * if the angle between the axes of joints 4 and 6 is smaller than
          TOL_WRIST, those axes are aligned and every solution returned has
          q4 = 0.
    """
    T_d = np.asarray(T_d, dtype=float)
    R_d = T_d[:3, :3]
    p_d = T_d[:3, 3]

    # Extrair a tabela DH modificada em q = 0: cada linha é [a_{i-1}, alpha_{i-1}, d_i, off_i]
    tbl = dh_table([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a = [float(tbl[i, 0]) for i in range(6)]
    alpha = [float(tbl[i, 1]) for i in range(6)]
    d = [float(tbl[i, 2]) for i in range(6)]
    off = [float(tbl[i, 3]) for i in range(6)]

    # Em MDH: ^{i-1}_i T(th_i) = T_const(a_{i-1}, alpha_{i-1}, d_i) @ RotZ(th_i)
    # onde T_const(a, alpha, d) = mdh_transform(a, alpha, d, 0.0).
    Tc = [mdh_transform(a[i], alpha[i], d[i], 0.0) for i in range(6)]

    # 1. Centro do punho (origem de {4}, que coincide com {5} pois a4=0, d5=0):
    # Como ^5_6 T(th6) = Tc[5] @ RotZ(th6), a origem de {6} em {5} é Tc[5][:3, 3] = [0, -d6*sin(alpha5), d6*cos(alpha5)],
    # que é paralela ao eixo Z_6 (3.ª coluna de Tc[5][:3, :3] é [0, -sin(alpha5), cos(alpha5)]).
    # Especificamente, no referencial {6}, a origem de {5} está em [0, 0, -d[5]]!
    # Logo, no referencial {0}: p_w = p_d - d[5] * R_d[:, 2]
    p_w = p_d - d[5] * R_d[:, 2]

    # 2. Equação para th1:
    # Em {1}, a posição do centro do punho p_w1 = RotZ(-th1) @ Tc[0]^{-1} @ [p_w; 1]
    # Sabemos que em {2}, a coordenada Z_2 de p_w é constante:
    # p_w em {2} é RotZ(th2) @ (Tc[2] @ RotZ(th3) @ Tc[3][: , 3]) -> a sua componente Z_2
    # não depende de th2 nem de th3 porque Z_2 é o eixo de rotação de th2 e alpha2 é 0 ou pi!
    # Especificamente, o vetor de {2} para {4} em {2} antes de rodar th2 tem componente Z_2 igual a:
    p34_in_3 = Tc[3][:, 3]  # [a3, -d4*sin(alpha3), d4*cos(alpha3), 1]
    # Como alpha2 é 0 ou pi (sin(alpha2) = 0), a linha Z de Tc[2] é [0, 0, cos(alpha2), d3*cos(alpha2)],
    # logo a coordenada Z_2 do centro do punho é constante:
    z2_const = float(Tc[2][2, :] @ p34_in_3)  # = 0 para o Lite 6

    # Por outro lado, p_w em {1} é p_w1 = RotZ(-th1) @ (Tc[0]^{-1} @ [p_w; 1]).
    # E p_w em {2} tem coordenada Z_2 dada pela 3.ª linha de Tc[1]^{-1} @ p_w1:
    # Como Tc[1][:3, :3] = Rx(alpha1), a 3.ª coluna de Rx(alpha1) é [0, -sin(alpha1), cos(alpha1)]^T.
    # Assim, Z_2 = -sin(alpha1) * y_1 + cos(alpha1) * (z_1 - d2) = z2_const.
    # Como alpha1 = +-pi/2 (cos(alpha1) = 0) e y_1 = -x0*sin(th1) + y0*cos(th1):
    p_w0 = np.linalg.inv(Tc[0]) @ np.append(p_w, 1.0)
    x0, y0, z0 = float(p_w0[0]), float(p_w0[1]), float(p_w0[2])

    # Se x0 == 0 e y0 == 0 (singularidade do ombro), th1 = 0
    if math.hypot(x0, y0) < 1e-12:
        th1_candidates = [0.0, math.pi]
    else:
        th1_base = math.atan2(y0, x0)
        th1_candidates = [th1_base, th1_base + math.pi]

    raw_solutions = []

    for th1 in th1_candidates:
        q1 = th1 - off[0]

        # Posição do centro do punho expressa no referencial intermédio antes de rodar th2:
        # Seja T01 = mdh_transform(a[0], alpha[0], d[0], th1)
        # e Tc1 = Tc[1] = mdh_transform(a[1], alpha[1], d[1], 0.0).
        # Então no referencial {2_zero} (isto é, {2} com th2 = 0):
        T0_2zero = mdh_transform(a[0], alpha[0], d[0], th1) @ Tc[1]
        pw_2zero = np.linalg.inv(T0_2zero) @ np.append(p_w, 1.0)
        rx, ry = float(pw_2zero[0]), float(pw_2zero[1])

        # No referencial {2_zero}, pw_2zero = RotZ(th2) @ Tc[2] @ RotZ(th3) @ p34_in_3.
        # Seja v(th3) = Tc[2] @ RotZ(th3) @ p34_in_3.
        # Em função de cos(th3) e sin(th3):
        # RotZ(th3) @ p34_in_3 = [p34_x * c3 - p34_y * s3, p34_x * s3 + p34_y * c3, p34_z, 1]^T
        p34_x, p34_y = float(p34_in_3[0]), float(p34_in_3[1])
        ca2 = math.cos(alpha[2])  # +1 ou -1

        # v_x(th3) = a[2] + p34_x * c3 - p34_y * s3
        # v_y(th3) = ca2 * (p34_x * s3 + p34_y * c3)
        # Como [rx, ry]^T = RotZ(th2) @ [v_x, v_y]^T, temos:
        # rx^2 + ry^2 = v_x^2 + v_y^2 = a[2]^2 + (p34_x^2 + p34_y^2) + 2*a[2]*(p34_x * c3 - p34_y * s3)
        L_sq = p34_x * p34_x + p34_y * p34_y
        rhs = rx * rx + ry * ry - a[2] * a[2] - L_sq
        th3_list = _solve_trig(2.0 * a[2] * p34_x, -2.0 * a[2] * p34_y, rhs)

        for th3 in th3_list:
            q3 = th3 - off[2]
            c3, s3 = math.cos(th3), math.sin(th3)
            vx = a[2] + p34_x * c3 - p34_y * s3
            vy = ca2 * (p34_x * s3 + p34_y * c3)

            # [rx, ry]^T = [vx * c2 - vy * s2, vx * s2 + vy * c2]^T
            th2 = math.atan2(vx * ry - vy * rx, vx * rx + vy * ry)
            q2 = th2 - off[1]

            # Calcular ^0_3 T e isolar a rotação do punho:
            t01 = mdh_transform(a[0], alpha[0], d[0], th1)
            t12 = mdh_transform(a[1], alpha[1], d[1], th2)
            t23 = mdh_transform(a[2], alpha[2], d[2], th3)
            R03 = (t01 @ t12 @ t23)[:3, :3]

            # Queremos resolver R03 @ R34(th4) @ R45(th5) @ R56(th6) = R_d
            # Passando a rotação constante de Tc[3] para a esquerda e a de Tc[5] para a direita:
            # R34(th4) = Rc3 @ RotZ(th4), R45(th5) = Rc4 @ RotZ(th5), R56(th6) = Rc5 @ RotZ(th6)
            # Logo: RotZ(th4) @ Rc4 @ RotZ(th5) @ Rc5 @ RotZ(th6) = (R03 @ Rc3)^T @ R_d = M
            Rc3 = Tc[3][:3, :3]
            Rc4 = Tc[4][:3, :3]
            Rc5 = Tc[5][:3, :3]
            M = (R03 @ Rc3).T @ R_d

            # Além disso, podemos separar Rc5 = Rx(alpha5):
            # Seja W = M @ Rc5^T = RotZ(th4) @ Rc4 @ RotZ(th5) @ (Rc5 @ RotZ(th6) @ Rc5^T).
            # Mas é ainda mais simples: o eixo da junta 4 no referencial após Rc3 é Z = [0, 0, 1]^T.
            # E o eixo da junta 6 é a 3.ª coluna de M? Não: o eixo da junta 6 é o eixo Z do referencial {6},
            # que é invariante a RotZ(th6)! Logo, a 3.ª coluna de M é:
            # M[:, 2] = RotZ(th4) @ Rc4 @ RotZ(th5) @ Rc5[:, 2].
            # E a 3.ª linha de (Rc5^T @ M^T), isto é, (Rc5[:, 2])^T @ M^T = M @ ... espera:
            # Como RotZ(th6) deixa [0, 0, 1]^T invariante: RotZ(th6) @ [0, 0, 1]^T = [0, 0, 1]^T!
            # Logo, a 3.ª coluna de M é EXATAMENTE:
            # m_col2 = RotZ(th4) @ Rc4 @ RotZ(th5) @ Rc5 @ [0, 0, 1]^T!
            # E como [0, 0, 1] @ RotZ(th4) = [0, 0, 1], a 3.ª linha de M é:
            # m_row2 = [0, 0, 1] @ Rc4 @ RotZ(th5) @ Rc5 @ RotZ(th6)!

            # No Lite 6, alpha4 = +-pi/2 e alpha5 = +-pi/2:
            sa4 = math.sin(alpha[4])  # -1 ou +1
            sa5 = math.sin(alpha[5])  # +1 ou -1
            # Rc5 @ [0, 0, 1]^T = [0, -sa5, 0]^T
            # RotZ(th5) @ [0, -sa5, 0]^T = [sa5 * sin(th5), -sa5 * cos(th5), 0]^T
            # Rc4 @ [sa5 * s5, -sa5 * c5, 0]^T = [sa5 * s5, 0, -sa4 * sa5 * c5]^T
            # RotZ(th4) @ [sa5 * s5, 0, -sa4 * sa5 * c5]^T =
            #   [sa5 * s5 * cos(th4), sa5 * s5 * sin(th4), -sa4 * sa5 * cos(th5)]^T = M[:, 2]!
            c_th5 = -sa4 * sa5 * float(M[2, 2])
            s_th5_mag = math.hypot(float(M[0, 2]), float(M[1, 2]))

            # Ângulo entre os eixos das juntas 4 e 6:
            # O eixo da junta 4 (no frame após Rc3) é [0, 0, 1]^T; o eixo da junta 6 é M[:, 2].
            # Portanto o ângulo entre eles tem seno = s_th5_mag e cosseno = |M[2, 2]| = |c_th5|.
            wrist_angle = math.atan2(s_th5_mag, abs(c_th5))

            if wrist_angle < TOL_WRIST:
                q4 = 0.0
                th4 = off[3]
                th5 = 0.0 if c_th5 >= 0.0 else math.pi
                q5 = th5 - off[4]

                # Isolar th6 diretamente de RotZ(th6) = (RotZ(th4) @ Rc4 @ RotZ(th5) @ Rc5)^T @ M
                R_w45 = mdh_transform(0.0, 0.0, 0.0, th4)[:3, :3] @ Rc4 @ mdh_transform(0.0, 0.0, 0.0, th5)[:3, :3] @ Rc5
                R_z6 = R_w45.T @ M
                th6 = math.atan2(float(R_z6[1, 0]), float(R_z6[0, 0]))
                q6 = th6 - off[5]
                raw_solutions.append([q1, q2, q3, q4, q5, q6])
            else:
                for sign_w in (1.0, -1.0):
                    s_th5 = sign_w * s_th5_mag
                    th5 = math.atan2(s_th5, c_th5)
                    q5 = th5 - off[4]

                    # M[0, 2] = sa5 * s_th5 * cos(th4)
                    # M[1, 2] = sa5 * s_th5 * sin(th4)
                    denom4 = sa5 * s_th5
                    th4 = math.atan2(float(M[1, 2]) / denom4, float(M[0, 2]) / denom4)
                    q4 = th4 - off[3]

                    # Com th4 e th5 determinados, RotZ(th6) = (RotZ(th4) @ Rc4 @ RotZ(th5) @ Rc5)^T @ M
                    c4, s4 = math.cos(th4), math.sin(th4)
                    c5, s5 = math.cos(th5), math.sin(th5)
                    # Primeira linha da matriz (RotZ(th4) @ Rc4 @ RotZ(th5) @ Rc5)^T é a 1.ª coluna transposta:
                    # 1.ª coluna de RotZ(th4)@Rc4@RotZ(th5)@Rc5 é [c4*c5, s4*c5, sa4*s5]^T
                    # 2.ª coluna de RotZ(th4)@Rc4@RotZ(th5)@Rc5 é [-c4*s5*ca5 + s4*sa4*sa5, ...] ->
                    # Usamos o produto de matrizes 3x3 explícito para precisão de máquina total:
                    Rz4 = np.array([[c4, -s4, 0.0], [s4, c4, 0.0], [0.0, 0.0, 1.0]])
                    Rz5 = np.array([[c5, -s5, 0.0], [s5, c5, 0.0], [0.0, 0.0, 1.0]])
                    R_z6 = (Rz4 @ Rc4 @ Rz5 @ Rc5).T @ M
                    th6 = math.atan2(float(R_z6[1, 0]), float(R_z6[0, 0]))
                    q6 = th6 - off[5]

                    raw_solutions.append([q1, q2, q3, q4, q5, q6])

    # Filtrar duplicados (diferença módulo 2*pi <= TOL_SAME) e normalizar para (-pi, pi]
    distinct_solutions = []
    for sol in raw_solutions:
        wrapped = np.array([_wrap_angle(ang) for ang in sol], dtype=float)
        is_duplicate = False
        for existing in distinct_solutions:
            diffs = np.abs((wrapped - existing + math.pi) % (2.0 * math.pi) - math.pi)
            if np.all(diffs <= TOL_SAME):
                is_duplicate = True
                break
        if not is_duplicate:
            distinct_solutions.append(wrapped)

    return distinct_solutions


def select_solution(solutions, q_current):
    """The solution that is executed, chosen among `solutions`.

    Args:
        solutions: list of (6,) joint vectors (rad), as inverse_kinematics()
                   returns them. Their angles may be any real numbers: this
                   function must not assume they are in (-pi, pi].
        q_current: six joint angles (rad), the current configuration of the arm.

    Returns:
        A (6,) float array, the selected joint vector, or None if no solution
        is executable (no solutions at all, or all discarded). The inputs are
        not modified.

    The rule is the one of Part A: in each solution, every angle q_i is
    replaced by the value q_i + 2 pi k (k an integer) that lies inside the
    range of joint i (JOINT_LIMITS) and is closest to q_current[i]; a solution
    for which no such value exists is discarded. Among the solutions that
    remain, the one at the smallest euclidean distance from q_current is
    selected, and the returned vector is that solution with the replaced angles.
    """
    if not solutions:
        return None

    q_curr = np.asarray(q_current, dtype=float)
    best_sol = None
    best_dist = float('inf')

    for sol in solutions:
        sol_arr = np.asarray(sol, dtype=float)
        candidate = np.empty(6, dtype=float)
        valid = True

        for i in range(6):
            low, high = JOINT_LIMITS[i]
            qi = sol_arr[i]
            k_min = math.ceil((low - qi) / (2.0 * math.pi) - 1e-12)
            k_max = math.floor((high - qi) / (2.0 * math.pi) + 1e-12)

            best_val = None
            min_diff = float('inf')
            for k in range(k_min, k_max + 1):
                val = qi + 2.0 * math.pi * k
                if low - 1e-12 <= val <= high + 1e-12:
                    val_clamped = max(low, min(high, val))
                    diff = abs(val_clamped - q_curr[i])
                    if diff < min_diff:
                        min_diff = diff
                        best_val = val_clamped

            if best_val is None:
                valid = False
                break
            candidate[i] = best_val

        if valid:
            dist = float(np.linalg.norm(candidate - q_curr))
            if dist < best_dist:
                best_dist = dist
                best_sol = candidate

    return best_sol


# =============================================================================
# Part B - the node
# =============================================================================
class IkNode(Node):
    """Solves the inverse kinematics of every desired pose it receives.

    Subscribes   target_topic         geometry_msgs/PoseStamped  (/lite6_ik/target)
                 joint_states_topic   sensor_msgs/JointState     (/joint_states)
    Publishes    solution_topic       sensor_msgs/JointState     (/lite6_ik/solution)

    The target is the pose of link_eef in link_base (header.frame_id is
    BASE_FRAME). For each target received, the node computes the solutions and
    selects one from the latest joint angles received. If there is a selected
    solution it publishes a JointState:
        name            JOINT_NAMES
        position        the selected solution, in the order of JOINT_NAMES
        header.stamp    the stamp of the TARGET it answers (copy it:
                        ik_check matches the solution to the target by it)
    Otherwise it publishes nothing and logs why:
        unreachable     the pose has no solution at all
        out_of_range    it has solutions, but all of them were discarded

    The topic names are constructor arguments so that the tests can run the
    node on private topics: use them, do not write the names again.
    """

    def __init__(self, target_topic='/lite6_ik/target', joint_states_topic='/joint_states',
                 solution_topic='/lite6_ik/solution'):
        super().__init__('ik_node')
        self.q_current = None
        self.publisher = self.create_publisher(JointState, solution_topic, 10)
        self.sub_joints = self.create_subscription(
            JointState, joint_states_topic, self.on_joint_state, 10
        )
        self.sub_target = self.create_subscription(
            PoseStamped, target_topic, self.on_target, 10
        )

    def on_joint_state(self, msg):
        """Called for every JointState: keep the latest joint angles."""
        try:
            self.q_current = np.array(
                [msg.position[msg.name.index(name)] for name in JOINT_NAMES],
                dtype=float
            )
        except ValueError:
            return

    def on_target(self, msg):
        """Called for every desired pose: solve it and publish the solution."""
        if self.q_current is None:
            self.get_logger().warning('No joint states received yet.')
            return

        T_d = transform_from_pose(msg)
        solutions = inverse_kinematics(T_d)

        if not solutions:
            self.get_logger().info('unreachable')
            print('unreachable', flush=True)
            return

        selected = select_solution(solutions, self.q_current)
        if selected is None:
            self.get_logger().info('out_of_range')
            print('out_of_range', flush=True)
            return

        out_msg = JointState()
        out_msg.header.stamp = msg.header.stamp
        out_msg.name = list(JOINT_NAMES)
        out_msg.position = [float(x) for x in selected]
        self.publisher.publish(out_msg)


# =============================================================================
# PROVIDED - DO NOT EDIT. Already implemented; nothing below is yours to change.
# =============================================================================
TOL_SAME = 1e-6     # rad - two solutions closer than this in every angle are the same
TOL_WRIST = 1e-6    # rad - angle between the axes of joints 4 and 6 below which they are aligned

# Joint ranges (rad) of the xarm_description model of the Lite 6, one row per
# joint, [lower, upper].
JOINT_LIMITS = np.array([
    [-math.pi * 0.99, math.pi * 0.99],
    [-2.61799, 2.61799],
    [-0.061087, math.pi * 0.99],
    [-math.pi * 0.99, math.pi * 0.99],
    [-2.1642, 2.1642],
    [-math.pi * 0.99, math.pi * 0.99],
])


def transform_from_pose(msg) -> np.ndarray:
    """(4, 4) transform of a geometry_msgs/PoseStamped.

    The quaternion is normalised first, so a slightly off unit quaternion still
    gives a rotation matrix.
    """
    p, o = msg.pose.position, msg.pose.orientation
    n = math.sqrt(o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w)
    x, y, z, w = o.x / n, o.y / n, o.z / n, o.w / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def main(args=None):
    """Entry point of ``ros2 run lite6_ik ik_node``."""
    if rclpy is None:
        raise SystemExit('ROS 2 is not available here: run this inside the container.')
    rclpy.init(args=args)
    node = None
    status = 0
    try:
        node = IkNode()
        node.get_logger().info('ik_node running: desired poses in, joint angles out')
        rclpy.spin(node)
    except NotImplementedError as e:
        print(f'\n{e}\nImplement it in lite6_ik/inverse_kinematics.py first.')
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