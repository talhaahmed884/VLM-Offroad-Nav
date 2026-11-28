"""Tests for the run videos: chase-camera geometry, the frame layout, the replay and the recorder (no MAVS)."""

import math

import numpy as np
import pytest
from test_sim_loop import CALIB, FlatWorld, straight_trail

from offroad.sim.choosers import ReplayChooser, ReplayMismatch, StraightestChooser
from offroad.sim.loop import LoopParams, run_episode
from offroad.sim.rig import yaw_from_quaternion
from offroad.sim.video import (FPS, ChaseParams, Recorder, answer_text, chase_K, chase_pose, compose_frame,
                               follow_yaw, project_to_chase, quat_matrix, quat_yaw_pitch)


def test_quaternions_turn_and_tilt_as_named():
    q = quat_yaw_pitch(math.radians(30), 0.0)
    assert yaw_from_quaternion(q) == pytest.approx(math.radians(30))
    forward = quat_matrix(quat_yaw_pitch(0.0, math.radians(20))) @ np.array([1.0, 0.0, 0.0])
    assert forward[2] == pytest.approx(-math.sin(math.radians(20)))  # + pitch looks down


def test_chase_camera_sees_the_vehicle_ahead_of_it():
    params = ChaseParams()
    vehicle = np.array([10.0, 5.0, 0.0])
    position, quat = chase_pose(vehicle, math.pi / 2, params)  # heading north: camera to the south
    assert position == pytest.approx([10.0, 5.0 - params.back_m, params.up_m])
    w, h = params.size
    # A point on the camera's optical axis lands in the image centre; one to the left of it, left of the centre.
    axis = np.array(position) + quat_matrix(quat) @ np.array([12.0, 0.0, 0.0])
    left = np.array(position) + quat_matrix(quat) @ np.array([12.0, 2.0, 0.0])
    behind = np.array(position) + quat_matrix(quat) @ np.array([-3.0, 0.0, 0.0])
    uv = project_to_chase(np.vstack([axis, left, behind, vehicle]), position, quat, params)
    assert uv[0] == pytest.approx([w / 2, h / 2])
    assert uv[1, 0] < w / 2
    assert np.isnan(uv[2]).all()
    assert 0 < uv[3, 0] < w and h / 2 < uv[3, 1] < h  # the vehicle: in view, below the centre
    assert chase_K(params)[0, 0] == pytest.approx((w / 2) / math.tan(math.radians(params.fov_deg / 2)))


def test_follow_yaw_takes_the_short_way_round():
    assert follow_yaw(None, 1.0, 0.2) == 1.0
    assert follow_yaw(math.radians(170), math.radians(-170), 0.5) == pytest.approx(math.radians(180))


def test_frame_layout():
    chase = np.zeros((540, 960, 3), np.uint8)
    picture = np.full((600, 800, 3), 200, np.uint8)
    frame = compose_frame(chase, picture, "what the VLM saw", ["header", "line 1", "line 2"])
    assert frame.shape[1] == 960 + 720 and frame.shape[0] % 2 == 0 and frame.shape[0] > 540
    assert compose_frame(chase, None, "t", ["x"]).shape[1] == 960 + 720
    record = {"n_candidates": 2, "chosen": 1, "chosen_bearing_deg": -3.0}
    assert "blue" in answer_text(record, {"colors": ["red", "blue"]})
    vlm = {"valid": True, "choice": "red", "safe_until": {"red": "all"}, "action": "continue", "seconds": 4.2}
    assert "answer: red" in answer_text(record, vlm) and "4.2 s" in answer_text(record, vlm)
    assert "keeps the previous path" in answer_text({"n_candidates": 0, "held": True}, {})


def test_a_replay_drives_the_saved_run_again():
    saved = run_episode(FlatWorld(straight_trail(25.0)), StraightestChooser(), CALIB, "flat",
                        params=LoopParams(start_waypoint=0))
    replay = run_episode(FlatWorld(straight_trail(25.0)), ReplayChooser(saved.decisions, "straightest", CALIB),
                         CALIB, "flat", params=LoopParams(start_waypoint=0))
    assert replay.outcome == saved.outcome and len(replay.decisions) == len(saved.decisions)
    assert np.allclose(replay.trajectory, saved.trajectory)
    changed = [dict(d) for d in saved.decisions]
    changed[3]["end_bearings_deg"] = [b + 1.0 for b in changed[3]["end_bearings_deg"]]
    with pytest.raises(ReplayMismatch):
        run_episode(FlatWorld(straight_trail(25.0)), ReplayChooser(changed, "straightest", CALIB), CALIB, "flat",
                    params=LoopParams(start_waypoint=0))


class FilmedFlatWorld(FlatWorld):
    def chase_view(self, camera_position, camera_quat):
        return np.zeros((540, 960, 3), np.uint8)

    def ground_height(self, x, y):
        return 0.0


def test_the_recorder_writes_a_frame_every_tenth_of_a_simulated_second():
    frames = []
    saved = run_episode(FlatWorld(straight_trail(25.0)), StraightestChooser(), CALIB, "flat",
                        params=LoopParams(start_waypoint=0))
    recorder = Recorder(FilmedFlatWorld(straight_trail(25.0)), frames.append, "header")
    ep = run_episode(recorder, ReplayChooser(saved.decisions, "straightest", CALIB), CALIB, "flat",
                     params=LoopParams(start_waypoint=0), on_decision=recorder.on_decision)
    recorder.finish("finished", hold_s=1.0)
    sim_s = ep.summary["sim_time_s"]
    assert len(frames) == recorder.frames == 1 + round(sim_s * FPS) + FPS  # first view + the drive + the hold
    assert recorder.path_world is not None and recorder.picture is not None
    assert frames[-1].shape == frames[0].shape
