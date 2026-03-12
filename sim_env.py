"""
Raw MuJoCo environment for SO-ARM100 pick-and-place.
No gymnasium dependency — direct MuJoCo Python bindings.

Uses a kinematic grasp model: when the gripper is close enough to the cube
and the jaw is closed, the cube is attached to the gripper (position locked).
This is standard for generating demonstration trajectories in sim.
"""

import argparse
import pathlib
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import Image

import config


@dataclass
class EpisodeResult:
    frames: list = field(default_factory=list)  # list of numpy RGB arrays
    success: bool = False
    instruction: str = ""
    num_steps: int = 0
    cube_initial_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bin_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _load_model(scene_xml: str) -> mujoco.MjModel:
    """Load scene XML with gripper_tip site injected into the SO-ARM100 model."""
    scene_path = pathlib.Path(scene_xml)
    scene_text = scene_path.read_text()
    so_arm_path = scene_path.parent / "mujoco_menagerie" / "trs_so_arm100" / "so_arm100.xml"
    so_arm_text = so_arm_path.read_text()

    # Inject gripper_tip site into Fixed_Jaw body (before Moving_Jaw)
    # pos y=-0.065: center of the finger pad grasping zone (pads span y=-0.035 to y=-0.101)
    site_xml = '                <site name="gripper_tip" pos="0 -0.065 0" size="0.005" rgba="0 1 0 0.5"/>'
    so_arm_text = so_arm_text.replace(
        '<body name="Moving_Jaw"',
        site_xml + "\n                " + '<body name="Moving_Jaw"',
    )
    # Fix meshdir to be relative to project root
    so_arm_text = so_arm_text.replace(
        'meshdir="assets/"',
        'meshdir="mujoco_menagerie/trs_so_arm100/assets/"',
    )
    # Extract inner content (between <mujoco> and </mujoco>)
    match = re.search(r"<mujoco[^>]*>(.*?)</mujoco>", so_arm_text, re.DOTALL)
    inner = match.group(1)
    # Increase actuator strength so arm can reach low positions against gravity
    so_arm_text_mod = inner.replace(
        'kp="50" dampratio="1" forcerange="-3.5 3.5"',
        'kp="2000" dampratio="1" forcerange="-100 100"',
    )
    # Replace include with inline content
    combined = scene_text.replace(
        '<include file="mujoco_menagerie/trs_so_arm100/so_arm100.xml"/>',
        so_arm_text_mod,
    )
    return mujoco.MjModel.from_xml_string(combined)


class SO100PickPlaceEnv:
    """MuJoCo environment for SO-ARM100 pick-and-place."""

    # Default cube and bin positions (from scene.xml)
    DEFAULT_CUBE_POS = np.array([0.0, -0.22, 0.015])
    DEFAULT_BIN_POS = np.array([0.10, -0.32, 0.0])

    # Home joint configuration (from menagerie keyframe)
    HOME_QPOS = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])

    # Grasp parameters
    GRASP_DIST_THRESHOLD = 0.08  # gripper must be within 8cm of cube to grasp
    GRIPPER_OPEN_VAL = 1.5       # Jaw joint value for open
    GRIPPER_CLOSED_VAL = -0.174  # Jaw joint value for closed
    GRIPPER_CLOSED_THRESHOLD = 0.2  # Jaw < this = closed

    def __init__(self, scene_xml=None):
        scene_xml = scene_xml or str(config.SCENE_XML)
        self.model = _load_model(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, config.SIM_RENDER_HEIGHT, config.SIM_RENDER_WIDTH
        )

        # Cache IDs
        self._cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config.CUBE_BODY_NAME
        )
        self._bin_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config.BIN_BODY_NAME
        )
        self._gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, config.GRIPPER_SITE_NAME
        )

        # Freejoint qpos address for cube (3 pos + 4 quat = 7 values)
        cube_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self._cube_qpos_adr = self.model.jnt_qposadr[cube_jnt_id]
        self._cube_qvel_adr = self.model.jnt_dofadr[cube_jnt_id]

        self.frames = []
        self._grasped = False
        self._grasp_offset = np.zeros(3)  # cube offset relative to gripper when grasped

    def reset(self, cube_pos=None, bin_pos=None) -> np.ndarray:
        """Reset environment. Returns first rendered frame."""
        mujoco.mj_resetData(self.model, self.data)

        # Set arm to home position
        self.data.qpos[:6] = self.HOME_QPOS
        self.data.ctrl[:6] = self.HOME_QPOS

        # Set cube position via freejoint qpos
        cp = cube_pos if cube_pos is not None else self.DEFAULT_CUBE_POS
        adr = self._cube_qpos_adr
        self.data.qpos[adr : adr + 3] = cp
        self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]  # identity quat

        mujoco.mj_forward(self.model, self.data)

        # Let the scene settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.frames = []
        self._grasped = False
        self._grasp_offset = np.zeros(3)
        return self.render()

    def step(self, ctrl: np.ndarray) -> dict:
        """Apply control and step simulation."""
        self.data.ctrl[:6] = ctrl[:6]

        for _ in range(config.CTRL_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            # Kinematic grasp: if grasped, lock cube to gripper
            if self._grasped:
                self._attach_cube_to_gripper()

        gripper_pos = self.data.site_xpos[self._gripper_site_id].copy()
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        bin_pos = self.data.xpos[self._bin_body_id].copy()

        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_to_bin = np.linalg.norm(cube_pos[:2] - bin_pos[:2])
        cube_in_bin = cube_to_bin < 0.04 and cube_pos[2] > bin_pos[2]

        return {
            "cube_pos": cube_pos,
            "bin_pos": bin_pos,
            "gripper_pos": gripper_pos,
            "gripper_to_cube_dist": gripper_to_cube,
            "cube_to_bin_dist": cube_to_bin,
            "cube_in_bin": cube_in_bin,
        }

    def _attach_cube_to_gripper(self):
        """Lock cube position relative to gripper tip."""
        gripper_pos = self.data.site_xpos[self._gripper_site_id].copy()
        target_pos = gripper_pos + self._grasp_offset
        adr = self._cube_qpos_adr
        self.data.qpos[adr : adr + 3] = target_pos
        # Zero out cube velocity
        vadr = self._cube_qvel_adr
        self.data.qvel[vadr : vadr + 6] = 0

    def grasp(self):
        """Attempt to grasp: if gripper close to cube, snap cube to gripper center."""
        gripper_pos = self.data.site_xpos[self._gripper_site_id].copy()
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        dist = np.linalg.norm(gripper_pos - cube_pos)
        if dist < self.GRASP_DIST_THRESHOLD:
            self._grasped = True
            # Zero offset: cube center snaps to gripper_tip site position
            self._grasp_offset = np.zeros(3)
            return True
        return False

    def release(self):
        """Release the cube."""
        self._grasped = False
        self._grasp_offset = np.zeros(3)

    def render(self, camera_name=None) -> np.ndarray:
        """Render frame from camera, append to self.frames, return RGB array."""
        camera_name = camera_name or config.SIM_CAMERA_NAME
        self.renderer.update_scene(self.data, camera_name)
        img = self.renderer.render()
        self.frames.append(img.copy())
        return img

    def get_joint_positions(self) -> dict:
        """Return current joint angles as dict."""
        return {name: float(self.data.qpos[i]) for i, name in enumerate(config.JOINT_NAMES)}

    def get_frames_as_pil(self) -> list:
        """Convert captured frames to PIL Images."""
        return [Image.fromarray(f) for f in self.frames]

    def _interpolate(self, start: np.ndarray, end: np.ndarray, steps: int):
        """Generate linearly interpolated waypoints."""
        for i in range(steps):
            t = (i + 1) / steps
            yield start + t * (end - start)

    def _simple_ik_to_target(self, target_pos: np.ndarray, gripper_val: float = 0.0,
                              wrist_roll: float | None = None) -> np.ndarray:
        """
        Jacobian-based IK to find joint angles reaching target_pos.
        Uses iterative damped least squares. Does not step the simulation.

        Args:
            wrist_roll: If set, use full 5-DoF IK but regularize Wrist_Roll
                        toward this value. Use 0.0 for top-down grasps.
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        q = self.data.qpos[:5].copy()
        if wrist_roll is not None:
            q[4] = wrist_roll  # initialize at desired orientation

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        for _ in range(150):
            self.data.qpos[:5] = q
            mujoco.mj_forward(self.model, self.data)
            current = self.data.site_xpos[self._gripper_site_id].copy()
            error = target_pos - current

            if np.linalg.norm(error) < 0.003:
                break

            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._gripper_site_id)
            J = jac_pos[:, :5]

            lam = 0.05
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), error)
            q += 0.5 * dq

            # Soft regularization: pull wrist_roll toward desired value
            if wrist_roll is not None:
                q[4] += 0.1 * (wrist_roll - q[4])

            for i in range(5):
                lo = self.model.jnt_range[i, 0]
                hi = self.model.jnt_range[i, 1]
                q[i] = np.clip(q[i], lo, hi)

        # Restore state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)

        result = np.zeros(6)
        result[:5] = q
        result[5] = gripper_val
        return result

    def _apply_ctrl(self, ctrl: np.ndarray, noise_std: float = 0.0):
        """Apply control with optional noise and step."""
        if noise_std > 0:
            ctrl = ctrl.copy() + np.random.normal(0, noise_std, size=ctrl.shape)
            for i in range(min(len(ctrl), 6)):
                lo = self.model.jnt_range[i, 0]
                hi = self.model.jnt_range[i, 1]
                ctrl[i] = np.clip(ctrl[i], lo, hi)
        # Set both ctrl (for actuators) and qpos directly (for precise tracking)
        self.data.ctrl[:6] = ctrl[:6]
        self.data.qpos[:6] = ctrl[:6]
        for _ in range(config.CTRL_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if self._grasped:
                self._attach_cube_to_gripper()

    def run_scripted_episode(self, policy="pick_place", randomize=True) -> EpisodeResult:
        """Run a scripted episode and return results."""
        cube_pos = self.DEFAULT_CUBE_POS.copy()
        if randomize and policy != "random":
            cube_pos[0] += np.random.uniform(-0.03, 0.03)
            cube_pos[1] += np.random.uniform(-0.03, 0.03)

        self.reset(cube_pos=cube_pos)
        initial_cube_pos = self.data.xpos[self._cube_body_id].copy()
        bin_pos = self.data.xpos[self._bin_body_id].copy()

        if policy == "random":
            return self._run_random_policy(initial_cube_pos, bin_pos)
        elif policy == "noisy_pick_place":
            return self._run_pick_place(initial_cube_pos, bin_pos, noise_std=0.05)
        else:
            return self._run_pick_place(initial_cube_pos, bin_pos, noise_std=0.0)

    def _run_pick_place(self, initial_cube_pos, bin_pos, noise_std=0.0) -> EpisodeResult:
        """Scripted pick-and-place with kinematic grasp."""
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        STEPS = 20
        RENDER_EVERY = 4  # render a frame every N control steps for smooth video

        def current_ctrl():
            return np.concatenate([self.data.qpos[:5].copy(), [self.data.ctrl[5]]])

        def move_to(target, grip, steps=STEPS, wrist_roll=None):
            ctrl = self._simple_ik_to_target(target, grip, wrist_roll=wrist_roll)
            for i, c in enumerate(self._interpolate(current_ctrl(), ctrl, steps)):
                self._apply_ctrl(c, noise_std)
                if (i + 1) % RENDER_EVERY == 0:
                    self.render()
            # Hold final target to let actuators settle
            for _ in range(15):
                self._apply_ctrl(ctrl, noise_std)
            self.render()

        # Phase 1: Home with gripper open
        home = np.concatenate([self.HOME_QPOS[:5], [self.GRIPPER_OPEN_VAL]])
        for i, c in enumerate(self._interpolate(current_ctrl(), home, 10)):
            self._apply_ctrl(c, noise_std)
            if (i + 1) % RENDER_EVERY == 0:
                self.render()
        self.render()

        # Phase 2: Pre-grasp — above cube (wrist_roll=0 aligns jaws for top-down grasp)
        pre = cube_pos.copy()
        pre[2] += 0.08
        move_to(pre, self.GRIPPER_OPEN_VAL, wrist_roll=0.0)

        # Phase 3: Approach — at cube level
        approach = cube_pos.copy()
        approach[2] += 0.005
        move_to(approach, self.GRIPPER_OPEN_VAL, wrist_roll=0.0)

        # Phase 4: Grasp — close gripper + kinematic attach
        grasped = self.grasp()
        close_ctrl = current_ctrl()
        close_ctrl[5] = self.GRIPPER_CLOSED_VAL
        for i, c in enumerate(self._interpolate(current_ctrl(), close_ctrl, 10)):
            self._apply_ctrl(c, noise_std)
            if (i + 1) % RENDER_EVERY == 0:
                self.render()
        self.render()

        # Phase 5: Lift
        lift = cube_pos.copy()
        lift[2] += 0.12
        move_to(lift, self.GRIPPER_CLOSED_VAL, wrist_roll=0.0)

        # Phase 6: Over bin
        over_bin = bin_pos.copy()
        over_bin[2] += 0.12
        move_to(over_bin, self.GRIPPER_CLOSED_VAL, steps=25, wrist_roll=0.0)

        # Phase 7: Lower into bin
        lower = bin_pos.copy()
        lower[2] += 0.03
        move_to(lower, self.GRIPPER_CLOSED_VAL, wrist_roll=0.0)

        # Phase 8: Release
        self.release()
        open_ctrl = current_ctrl()
        open_ctrl[5] = self.GRIPPER_OPEN_VAL
        for i, c in enumerate(self._interpolate(current_ctrl(), open_ctrl, 10)):
            self._apply_ctrl(c, noise_std)
            if (i + 1) % RENDER_EVERY == 0:
                self.render()
        # Let cube settle after release
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.render()

        # Phase 9: Retract
        home = np.concatenate([self.HOME_QPOS[:5], [self.GRIPPER_OPEN_VAL]])
        for i, c in enumerate(self._interpolate(current_ctrl(), home, STEPS)):
            self._apply_ctrl(c, noise_std)
            if (i + 1) % RENDER_EVERY == 0:
                self.render()
        self.render()

        # Check success
        final_cube = self.data.xpos[self._cube_body_id].copy()
        cube_to_bin_xy = np.linalg.norm(final_cube[:2] - bin_pos[:2])
        success = cube_to_bin_xy < 0.04 and final_cube[2] > bin_pos[2] - 0.01

        return EpisodeResult(
            frames=list(self.frames),
            success=success,
            instruction=config.TASK_INSTRUCTION,
            num_steps=len(self.frames),
            cube_initial_pos=initial_cube_pos,
            bin_pos=bin_pos,
        )

    def _run_random_policy(self, initial_cube_pos, bin_pos) -> EpisodeResult:
        """Random joint targets — mostly fails."""
        for _ in range(20):
            ctrl = np.zeros(6)
            for i in range(6):
                lo = self.model.jnt_range[i, 0]
                hi = self.model.jnt_range[i, 1]
                ctrl[i] = np.random.uniform(lo, hi)
            for _ in range(config.CTRL_SUBSTEPS * 5):
                self.data.ctrl[:6] = ctrl
                mujoco.mj_step(self.model, self.data)
            self.render()

        final_cube = self.data.xpos[self._cube_body_id].copy()
        cube_to_bin_xy = np.linalg.norm(final_cube[:2] - bin_pos[:2])
        success = cube_to_bin_xy < 0.04 and final_cube[2] > bin_pos[2] - 0.01

        return EpisodeResult(
            frames=list(self.frames),
            success=success,
            instruction=config.TASK_INSTRUCTION,
            num_steps=len(self.frames),
            cube_initial_pos=initial_cube_pos,
            bin_pos=bin_pos,
        )

    def close(self):
        """Clean up renderer."""
        self.renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SO-ARM100 pick-and-place environment")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer (interactive)")
    args = parser.parse_args()

    if args.render:
        import mujoco.viewer
        import time as _time

        env = SO100PickPlaceEnv()
        env.reset()

        # Gather all waypoint controls for pick-and-place
        cube_pos = env.data.xpos[env._cube_body_id].copy()
        bin_pos = env.data.xpos[env._bin_body_id].copy()

        def current_ctrl():
            return np.concatenate([env.data.qpos[:5].copy(), [env.data.ctrl[5]]])

        OPEN = env.GRIPPER_OPEN_VAL
        CLOSED = env.GRIPPER_CLOSED_VAL
        STEPS = 20

        # Build list of (ctrl_target, label, do_grasp, do_release) actions
        actions = []

        # Home
        home = np.concatenate([env.HOME_QPOS[:5], [OPEN]])
        actions.append((home, "Home", False, False))

        # Pre-grasp (wrist_roll=0 aligns jaws for top-down grasp)
        pre = cube_pos.copy(); pre[2] += 0.08
        actions.append((env._simple_ik_to_target(pre, OPEN, wrist_roll=0.0), "Pre-grasp", False, False))

        # Approach
        approach = cube_pos.copy(); approach[2] += 0.005
        actions.append((env._simple_ik_to_target(approach, OPEN, wrist_roll=0.0), "Approach", False, False))

        # Grasp
        grasp_ctrl = env._simple_ik_to_target(approach, CLOSED, wrist_roll=0.0)
        actions.append((grasp_ctrl, "Grasp", True, False))

        # Lift
        lift = cube_pos.copy(); lift[2] += 0.12
        actions.append((env._simple_ik_to_target(lift, CLOSED, wrist_roll=0.0), "Lift", False, False))

        # Over bin
        over_bin = bin_pos.copy(); over_bin[2] += 0.12
        actions.append((env._simple_ik_to_target(over_bin, CLOSED, wrist_roll=0.0), "Over bin", False, False))

        # Lower
        lower = bin_pos.copy(); lower[2] += 0.03
        actions.append((env._simple_ik_to_target(lower, CLOSED, wrist_roll=0.0), "Lower", False, False))

        # Release
        release_ctrl = env._simple_ik_to_target(lower, OPEN, wrist_roll=0.0)
        actions.append((release_ctrl, "Release", False, True))

        # Retract
        actions.append((home, "Retract", False, False))

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while True:
                # Reset
                env.reset()
                print("Starting pick-and-place... (close viewer to exit)")

                for target, label, do_grasp, do_release in actions:
                    if not viewer.is_running():
                        break
                    print(f"  {label}")

                    if do_grasp:
                        env.grasp()
                    if do_release:
                        env.release()

                    start = current_ctrl()
                    for c in env._interpolate(start, target, STEPS):
                        if not viewer.is_running():
                            break
                        env.data.ctrl[:6] = c[:6]
                        for _ in range(config.CTRL_SUBSTEPS):
                            mujoco.mj_step(env.model, env.data)
                            if env._grasped:
                                env._attach_cube_to_gripper()
                        viewer.sync()
                        _time.sleep(0.02)  # ~50fps visual

                if not viewer.is_running():
                    break

                # Pause before looping
                print("  Done! Restarting in 2s...")
                _time.sleep(2.0)

        env.close()
    else:
        env = SO100PickPlaceEnv()

        policies = ["pick_place", "noisy_pick_place", "random"]
        for pol in policies:
            print(f"\n{'='*50}")
            print(f"Running policy: {pol}")
            result = env.run_scripted_episode(policy=pol, randomize=(pol != "random"))
            print(f"  Steps: {result.num_steps}")
            print(f"  Frames: {len(result.frames)}")
            print(f"  Success: {result.success}")
            print(f"  Cube start: {result.cube_initial_pos}")
            print(f"  Bin pos: {result.bin_pos}")

            if result.frames:
                Image.fromarray(result.frames[0]).save(f"/tmp/so100_{pol}_first.jpg")
                Image.fromarray(result.frames[-1]).save(f"/tmp/so100_{pol}_last.jpg")
                print(f"  Saved frames to /tmp/so100_{pol}_*.jpg")

        env.close()
        print("\nDone!")
