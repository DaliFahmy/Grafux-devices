"""
config.py
Central configuration for the Mitsubishi MELFA ROS2 agent.

All topic names, service names, action names, joint names, and planner
identifiers live here.  Import this module everywhere — no magic strings
in other files.
"""

# ---------------------------------------------------------------------------
# Robot kinematics
# ---------------------------------------------------------------------------

# Joint names exactly as declared in the MELFA URDF / SRDF.
# Override at runtime via CLI flag --joint-names if your URDF differs.
JOINT_NAMES: list[str] = ["j1", "j2", "j3", "j4", "j5", "j6"]

# Safe "home" configuration — all joints at zero degrees.
HOME_JOINTS_DEG: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ---------------------------------------------------------------------------
# MoveIt2 group / frame defaults
# ---------------------------------------------------------------------------

DEFAULT_MOVE_GROUP: str = "melfa_arm"
DEFAULT_BASE_LINK: str = "base_link"
DEFAULT_EE_LINK: str = "link_6"

# ---------------------------------------------------------------------------
# ROS2 topic names
# ---------------------------------------------------------------------------

TOPIC_JOINT_STATES: str = "/joint_states"       # sensor_msgs/msg/JointState
TOPIC_ROBOT_STATE: str = "/robot_state"          # moveit_msgs/msg/RobotState (optional)

# ---------------------------------------------------------------------------
# ROS2 service names
# ---------------------------------------------------------------------------

SERVICE_COMPUTE_FK: str = "/compute_fk"          # moveit_msgs/srv/GetPositionFK

# ---------------------------------------------------------------------------
# ROS2 action names
# ---------------------------------------------------------------------------

ACTION_MOVE_GROUP: str = "/move_group"            # moveit_msgs/action/MoveGroup
ACTION_MOVE_GROUP_TYPE: str = "moveit_msgs/action/MoveGroup"

# ---------------------------------------------------------------------------
# MoveIt2 planner identifiers
# ---------------------------------------------------------------------------

# Joint-space planning — uses OMPL (default MoveIt2 planner)
PLANNER_PIPELINE_JOINT: str = "ompl"
PLANNER_ID_JOINT: str = ""  # empty = use pipeline default

# Cartesian linear moves — uses Pilz Industrial Motion Planner
PLANNER_PIPELINE_CARTESIAN: str = "pilz_industrial_motion_planner"
PLANNER_ID_LIN: str = "LIN"  # Pilz Linear Cartesian planner

# ---------------------------------------------------------------------------
# Timing / retry defaults
# ---------------------------------------------------------------------------

# How long to wait for MoveIt2 to plan and execute (seconds)
MOVE_TIMEOUT_S: float = 30.0

# How long to wait for a service call response (seconds)
SERVICE_TIMEOUT_S: float = 10.0

# MoveIt2 planning parameters
PLANNING_ATTEMPTS: int = 10
PLANNING_TIME_S: float = 5.0

# rosbridge reconnect delay (seconds, doubles each attempt up to MAX)
ROSBRIDGE_RECONNECT_DELAY_S: float = 2.0
ROSBRIDGE_RECONNECT_MAX_S: float = 30.0

# Grafux device server reconnect delay (seconds)
GRAFUX_RECONNECT_DELAY_S: float = 5.0

# ---------------------------------------------------------------------------
# MoveIt2 error codes (moveit_msgs/msg/MoveItErrorCodes)
# ---------------------------------------------------------------------------

MOVEIT_SUCCESS: int = 1
MOVEIT_ERROR_CODES: dict[int, str] = {
    1: "SUCCESS",
    -1: "FAILURE",
    -2: "PLANNING_FAILED",
    -3: "INVALID_MOTION_PLAN",
    -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -5: "CONTROL_FAILED",
    -6: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -7: "TIMED_OUT",
    -8: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -26: "CRASH",
    -27: "ABORT",
    -28: "INCOMPATIBLE_SENSORS",
    -29: "NO_IK_SOLUTION",
}


def moveit_error_string(code: int) -> str:
    """Return a human-readable string for a MoveIt2 error code."""
    return MOVEIT_ERROR_CODES.get(code, f"UNKNOWN_ERROR({code})")
