
a= -0.9
b = -0.9
XWJG_CONFIGS = {
    "panda_wristcam": {
        "agent_pos": [-0.2, -0.6, 0],
        "holder_half_size": 0.02,
        "goal_thresh": 0.025,
        "holder_spawn_half_size": 0.05,
        "holder_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [
            a,
            b,
            1,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [a, b, 0],
        "human_cam_eye_pos": [
            a,
            b,
            0.5,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [a+0.1, b+0.1, 0.35],
    },
    "fetch": {
        "agent_pos": [-0.2, -0.6, 0],
        "holder_half_size": 0.02,
        "goal_thresh": 0.025,
        "holder_spawn_half_size": 0.1,
        "holder_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [
            a,
            b,
            1,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [a, b, 0],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
}