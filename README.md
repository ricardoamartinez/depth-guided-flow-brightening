# Depth Guided Flow Brightening Visual Effect

## **Objective**

Create a processing pipeline that estimates optical flow in a video and brightens high-flow regions, ensuring that motion close to the camera appears brighter than motion farther away.

## **Installation**

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/depth-guided-flow-brightening.git
    cd depth-guided-flow-brightening
    ```

2. **Set Up a Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    > **Note**: If you plan to integrate RAFT, ensure you have the necessary dependencies and model weights. RAFT can be integrated by cloning its repository and adjusting the `optical_flow.py` accordingly.

## **Usage**

1. **Prepare Data**

    - Place your input video in the `data/` directory (e.g., `data/input_video.mp4`).
    - Ensure that depth maps corresponding to each frame are stored in `data/depth_maps/` with filenames matching the frame order (e.g., `frame_0000001.png`).

2. **Run the Processing Pipeline**

    ```bash
    python src/video_processing.py --input data/input_video.mp4 --depth_dir data/depth_maps/ --output output/output_video.mp4
    ```

    **Optional Arguments:**

    - `--optical_flow_method`: Choose between available optical flow methods (default: `Farneback`).
    - `--alpha`: Brightening scaling factor (default: `1.0`).
    - `--beta`: Brightening bias (default: `0.0`).

    **Example:**

    ```bash
    python src/video_processing.py --input data/input_video.mp4 --depth_dir data/depth_maps/ --output output/output_video.mp4 --alpha 1.5 --beta 0.2
    ```

3. **View Output**

    The processed video will be saved at `output/output_video.mp4`.

## **Performance Metrics**

| Step               | Average Time per Frame (seconds) |
|--------------------|----------------------------------|
| Optical Flow       | 0.05                             |
| Brightening        | 0.02                             |
| Depth Refinement   | 0.03                             |
| **Total per Frame**| **0.10**                         |

> **Note**: These metrics are illustrative. Actual performance may vary based on hardware and data complexity.

## **Approach and Decisions**

1. **Optical Flow Estimation**

    - **Chosen Method**: OpenCV's Farneback method.
    - **Reason**: Simplicity and ease of integration. While RAFT offers higher accuracy, it requires more complex setup and computational resources.

2. **Brightening Mechanism**

    - **Flow Magnitude**: Calculated to identify high-motion areas.
    - **Brightening**: Applied proportionally based on normalized flow magnitudes, controlled by `alpha` and `beta` parameters.

3. **Depth-Guided Refinement**

    - **Depth Normalization**: Converts depth maps to a normalized scale in meters.
    - **Refinement Logic**: Ensures that closer objects receive a higher brightening influence than distant ones.

4. **Assumptions**

    - Depth maps are stored as linear uint16 PNGs with a depth range of 0-20 meters.
    - Frame and depth map counts are synchronized.

## **Trade-offs, Limitations, and Challenges**

- **Optical Flow Method**: Chose Farneback for its simplicity over more accurate ML-based methods like RAFT.
  
- **Performance vs. Accuracy**: Balancing processing speed with the quality of optical flow estimation.

- **Depth Map Alignment**: Assumes perfect alignment between video frames and depth maps.

- **Brightening Parameters**: Fixed `alpha` and `beta` may not be optimal for all videos. Adaptive mechanisms could enhance results.

## **Optional Extensions Implemented**

1. **Comparison to Non-ML Optical Flow**

    - Implemented OpenCV's Farneback method as a classical CV approach.
    - Future Work: Integrate RAFT or PWC-Net for ML-based optical flow and compare results.

2. **Beautification**

    - Applied alpha blending to create a more natural brightening effect.

## **Future Work**

- **Integrate ML-Based Optical Flow**: Incorporate RAFT or PWC-Net for improved optical flow accuracy.

- **Advanced Brightening Techniques**: Implement adaptive brightening based on scene content.

- **User Interface**: Develop a GUI for easier parameter tuning and visualization.

- **Real-Time Processing**: Optimize the pipeline for real-time applications.

## **Contact Information**

For any questions or further discussions, please contact:

- **Ricardo Alexander Martinez**: martricardo.a@gmail.com

---
