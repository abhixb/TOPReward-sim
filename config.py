import pathlib

# Paths
PROJECT_DIR = pathlib.Path(__file__).parent
MENAGERIE_DIR = PROJECT_DIR / "mujoco_menagerie"
SO_ARM100_DIR = MENAGERIE_DIR / "trs_so_arm100"
SCENE_XML = PROJECT_DIR / "scene.xml"

# Sim
SIM_TIMESTEP = 0.002  # MuJoCo timestep (seconds)
SIM_MAX_STEPS = 2000  # Max sim steps per episode (~4 seconds)
SIM_RENDER_WIDTH = 320
SIM_RENDER_HEIGHT = 240
SIM_CAMERA_NAME = "overhead"  # Camera defined in scene.xml
CTRL_SUBSTEPS = 10  # Sim steps per control step (control at ~50Hz)

# SO-ARM100 joint/actuator names (in order, as defined in so_arm100.xml)
JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

# Task
TASK_INSTRUCTION = "Pick up the red cube and place it in the box"
CUBE_BODY_NAME = "cube"
BIN_BODY_NAME = "bin"
GRIPPER_SITE_NAME = "gripper_tip"  # site at the end-effector tip

# Scoring
NUM_EVAL_FRAMES = 20
MAX_FRAMES_PER_PREFIX = 32
SUCCESS_THRESHOLD = -15.0  # log-prob threshold for single "True" token; real Qwen3-VL scores ~-8 to -15
SUCCESS_LAST_N = 3

# Output
OUTPUT_DIR = "outputs"
