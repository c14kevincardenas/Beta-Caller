import numpy as np


class MovementDetector:
    def __init__(self, movement_threshold=0.05, stability_threshold=15):
        """
        Initialize movement detector.

        Args:
            movement_threshold (float): Percentage threshold for detecting movement.
            stability_threshold (int): Number of stable frames to confirm stability.
        """
        self.previous_keypoints = None
        self.current_state = "stopped"  # Can be "stopped" or "moving"
        self.stability_count = 0
        self.stability_threshold = stability_threshold
        self.movement_threshold = movement_threshold

    def update(self, current_keypoints):
        """
        Updates the movement detector state based on current keypoints.

        Args:
            current_keypoints (np.ndarray): Array of current keypoints, shape (n_keypoints, 2).

        Returns:
            str: The current state: "stopped", "moving", or "move_completed".
        """
        if self.previous_keypoints is None:
            self.previous_keypoints = current_keypoints
            return "stopped"

        # Compute Euclidean distances for keypoints
        distances = np.linalg.norm(current_keypoints - self.previous_keypoints, axis=1)

        # Detect if any keypoint exceeds the movement threshold
        significant_movement = np.any(distances > self.movement_threshold)

        if significant_movement:
            self.current_state = "moving"
            self.stability_count = 0  # Reset stability count while moving
        else:
            self.stability_count += 1
            if self.stability_count >= self.stability_threshold:
                if self.current_state == "moving":
                    self.current_state = "stopped"
                    self.previous_keypoints = current_keypoints
                    return "move_completed"
                else:
                    self.current_state = "stopped"

        self.previous_keypoints = current_keypoints
        return self.current_state
