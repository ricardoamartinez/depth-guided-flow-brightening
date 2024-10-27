# src/optical_flow.py

import sys
import os
import numpy as np
import cv2
import torch

# Add the project root directory and RAFT directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raft_root = os.path.join(project_root, 'RAFT')
sys.path.append(project_root)
sys.path.append(raft_root)

# Now import RAFT modules
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder

class OpticalFlowEstimator:
    def __init__(self, method: str = 'RAFT', model_path: str = None):
        """
        Initialize the OpticalFlowEstimator.

        Args:
            method (str): The optical flow estimation method. Default is 'RAFT'.
            model_path (str): Path to the RAFT model weights.
        """
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.method == 'RAFT':
            if model_path is None or not os.path.exists(model_path):
                raise ValueError("Model path for RAFT is invalid.")
            self.raft_model = self.load_raft_model(model_path)
        elif self.method == 'Farneback':
            self.farneback_params = dict(
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unsupported optical flow method: {self.method}")

        print(f"Using device: {self.device}")

    def load_raft_model(self, model_path):
        args = type('Args', (), {
            "small": False,
            "mixed_precision": True,  # Enable mixed precision for faster computation
        })()
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.module.to(self.device)
        model.eval()
        return model

    def estimate_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Estimate optical flow between two frames.

        Args:
            frame1 (np.ndarray): The first frame (grayscale or color).
            frame2 (np.ndarray): The second frame (grayscale or color).

        Returns:
            np.ndarray: Optical flow with shape (H, W, 2).
        """
        if self.method == 'RAFT':
            # Preprocess frames
            frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(self.device)
            frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(self.device)
            
            padder = InputPadder(frame1.shape)
            frame1, frame2 = padder.pad(frame1[None], frame2[None])
            
            # Estimate flow
            with torch.cuda.amp.autocast(enabled=True):  # Enable automatic mixed precision
                with torch.no_grad():
                    _, flow = self.raft_model(frame1, frame2, iters=20, test_mode=True)
            
            # Post-process flow
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            
            return flow
        elif self.method == 'Farneback':
            # Convert to grayscale if input is color image
            if len(frame1.shape) == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if len(frame2.shape) == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            return cv2.calcOpticalFlowFarneback(
                prev=frame1,
                next=frame2,
                flow=None,
                **self.farneback_params
            )

def compute_optical_flow(frame1, frame2, method='Farneback', **kwargs):
    estimator = OpticalFlowEstimator(method=method, **kwargs)
    return estimator.estimate_flow(frame1, frame2)
