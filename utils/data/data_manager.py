from collections import deque
import os
from PIL import Image
import cv2


class PreviousMoves:
    def __init__(self, save_dir="saved_moves"):
        # store recent frames and limb movements
        self.recent_frames = deque(maxlen=2)
        self.recent_limbs = deque(maxlen=2)

        # directory for saving completed moves
        self.save_dir = save_dir
        self.limb_mapping = {
            0: 'left_hand',
            1: 'right_hand',
            2: 'left_foot',
            3: 'right_foot'
        }

        # create folders for each limb if they don't exist
        self._initialize_folders()

    def _initialize_folders(self):
        """Create parent and child folders for saving moves."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for limb in self.limb_mapping.values():
            limb_dir = os.path.join(self.save_dir, limb)
            if not os.path.exists(limb_dir):
                os.makedirs(limb_dir)

    def add_frame(self, frame):
        """Add a frame to the deque."""
        self.recent_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def add_limb(self, limb):
        """Add a limb movement to the deque."""
        self.recent_limbs.append(limb)

    def pop_and_save(self):
        """
        Pop the oldest frame and save it to disk in the folder corresponding
        to the predicted limb.
        """
        if self.recent_frames:
            # pop the oldest frame
            frame_to_save = self.recent_frames.popleft()

            # determine the folder based on the predicted limb
            limb = self.recent_limbs[0]
            limb_label = self.limb_mapping[limb]
            save_path = os.path.join(self.save_dir, limb_label)

            # generate a unique filename
            frame_count = len(os.listdir(save_path))
            file_name = f"{frame_count + 1}.jpg"
            file_path = os.path.join(save_path, file_name)

            # save the frame
            self._save_frame_to_disk(frame_to_save, file_path)

    def _save_frame_to_disk(self, frame, file_path):
        """
        Save a frame (as an image) to the specified file path.
        """
        # assuming the frame is in a format compatible with PIL.Image
        image = Image.fromarray(frame)
        image.save(file_path)

    def get_recent_data(self):
        """
        Retrieve the current contents of the recent frames and limb history.
        """
        return list(self.recent_frames), list(self.recent_limbs)

    def reset(self):
        """
        Clear all stored frames and limb history.
        """
        self.recent_frames.clear()
        self.recent_limbs.clear()
