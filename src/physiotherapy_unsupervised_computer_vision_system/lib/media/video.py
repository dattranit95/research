import os
import logging
import cv2 as cv
from cv2.typing import MatLike


def extract_video_frames(video_path: str, frame_rate=1) -> tuple[list[MatLike], int, int, int]:
    """
    Extract frames from a video file at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        frame_rate (int): The rate at which to extract frames (default is 1, meaning every frame).

    Returns:
        list[MatLike]: A list of frames extracted from the video.
        int: The width of the video frames.
        int: The height of the video frames.
        int: FPS of the video.
    """
    print(f"Extracting frames from video [{video_path}] with frame rate [{frame_rate}]...")

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    # Open the video file for extraction
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file [{video_path}].")
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frames = []
    count = 0

    # Extract frames at the specified frame rate
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(frame)
        count += 1

    # Release the video capture object
    cap.release()

    print(f"Extracted [{len(frames)}] frames from the video.")
    return frames, width, height, fps
