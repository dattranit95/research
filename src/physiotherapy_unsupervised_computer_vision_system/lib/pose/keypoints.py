from typing import Optional
from cv2.typing import MatLike
import cv2 as cv
import mediapipe as mp
import pandas as pd


# Load MediaPipe Pose model
mp_pose = mp.solutions.pose

def extract_pose_keypoints(frame: MatLike) -> tuple[Optional[pd.DataFrame], MatLike]:
    """
    Extract pose keypoints from a video frame using MediaPipe Pose.

    Args:
        frame (MatLike): The video frame from which to extract pose keypoints.

    Returns:
        pd.DataFrame: A DataFrame containing the pose keypoints.
    """
    print("Extracting pose keypoints from the frame...")

    # Convert the frame to RGB format
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(frame_rgb)

    # Extract pose landmarks if available
    if results.pose_landmarks:
        # Extract keypoints and their coordinates
        keypoints = [
            {"i": i, "x": lm.x, "y": lm.y, "z": lm.z, "confidence": lm.visibility}
            for i, lm in enumerate(results.pose_landmarks.landmark)
        ]
        df = pd.DataFrame(keypoints)
        print(f"Extracted [{len(df)}] pose keypoints from the frame.")

        # Draw the pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame_rgb = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        return df, frame_rgb
    else:
        print("No pose keypoints detected.")
        return None
