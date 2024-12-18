# Depth-Guided Flow Brightening Visual Effect 🌊✨

## **Objective** 🎯

Create a processing pipeline that estimates optical flow in a video and brightens high-flow regions, ensuring that motion closer to the camera appears brighter than motion farther away.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Processing Steps](#processing-steps)
    - [1. Optical Flow Estimation](#1-optical-flow-estimation)
    - [2. Brightening High-Flow Regions](#2-brightening-high-flow-regions)
    - [3. Depth-Guided Brightening](#3-depth-guided-brightening)
    - [4. Exporting the Modified Video](#4-exporting-the-modified-video)
7. [Performance Metrics](#performance-metrics)
8. [Trade-offs, Assumptions, and Limitations](#trade-offs-assumptions-and-limitations)
9. [Optional Extensions](#optional-extensions)
10. [Contact](#contact)
11. [License](#license)

---

## **Overview** 🔍

This project implements a video processing pipeline that integrates optical flow and depth maps to dynamically highlight motion within video sequences. The primary goal is to enhance visual regions with significant movement, especially those closer to the camera.

## **Example Output** 📹

https://user-images.githubusercontent.com/300ad93f-55fb-4eb0-9c8c-9d2f4bf14f4b

<!-- Fallback image/link in case embed doesn't load -->
<video width="100%" controls>
  <source src="https://user-images.githubusercontent.com/300ad93f-55fb-4eb0-9c8c-9d2f4bf14f4b" type="video/mp4">
</video>

<!-- Remove the other options since we now have the proper GitHub URL -->

---

## **Project Structure** 📁

```
DepthGuidedFlowBrightening/
├── data/
│   ├── input_video.mp4                 # Source video to be processed
│   └── source_depth/                   # Directory containing per-frame depth maps (.png files)
├── outputs/
│   └── output_visualization.mp4        # Output video with brightened high-flow regions
├── src/
│   ├── __init__.py
│   ├── motion_analysis/
│   │   ├── __init__.py
│   │   ├── motion_analyzer.py          # Core processing class
│   │   └── person_segmentation.py      # Person segmentation module (if applicable)
│   └── scripts/
│       └── process_video.py             # Main script to execute processing pipeline
├── tests/
│   ├── __init__.py
│   ├── test_motion_analyzer.py         # Unit tests for MotionAnalyzer
│   └── test_person_segmentation.py     # Unit tests for PersonSegmenter
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── setup.py                            # (Optional) Package installation script
```

---

## **Installation** 💻

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ricardoamartinez/depth-guided-flow-brightening.git
    cd depth-guided-flow-brightening
    ```

2. **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** If you plan to use advanced models for person segmentation or optical flow (e.g., TensorFlow, PyTorch), ensure to include and install those dependencies.

---

## **Quick Start** 🚀

Run the pipeline with FastFlowNet (recommended):

```bash
python src/scripts/process_video.py --flow_method fastflownet
```

Or use Farneback method (CPU-friendly alternative):

```bash
python src/scripts/process_video.py --flow_method farneback
```

For more detailed usage and configuration options, see the [Usage](#usage) section below.

---

## **Usage** 🚀

1. **Prepare Input Files**

    - **Input Video:** Place your source video (e.g., `input_video.mp4`) in the `data/` directory.
    - **Depth Maps:** Ensure that per-frame depth maps are available in the `data/source_depth/` directory. The depth maps should correspond to each frame of the input video and be in a linear `uint16` PNG format.

2. **Run the Processing Script**

    ```bash
    python src/scripts/process_video.py \
        --input_video data/input_video.mp4 \
        --depth_folder data/source_depth \
        --output_video outputs/output_visualization.mp4 \
        --flow_threshold_norm 0.25 \
        --threshold_dark_norm 0.9 \
        --contrast_sensitivity_norm 1.0 \
        --flow_brightness 1.0 \
        --preset balanced
    ```

    **Optional Arguments:**

    - `--flow_threshold_norm`: Normalized flow threshold [0.0, 1.0]. Default: `0.25`
    - `--threshold_dark_norm`: Normalized depth threshold [0.0, 1.0]. Default: `0.9`
    - `--contrast_sensitivity_norm`: Normalized contrast sensitivity [0.0, 1.0]. Default: `1.0`
    - `--flow_brightness`: Controls brightness of optical flow visualization (0.5 to 3.0). Default: `1.0`
    - `--preset`: One of 'speed', 'balanced', 'quality', or 'max_quality'. Default: `'balanced'`

    **Example with Custom Parameters:**

    ```bash
    python src/scripts/process_video.py \
        --input_video data/input_video.mp4 \
        --depth_folder data/source_depth \
        --output_video outputs/output_visualization.mp4 \
        --flow_threshold_norm 0.5 \
        --threshold_dark_norm 0.7 \
        --contrast_sensitivity_norm 0.8 \
        --flow_brightness 2.0 \
        --preset quality
    ```

3. **Output**

    The processed video will be saved in the `outputs/` directory as `output_visualization.mp4`, showcasing brightened high-flow regions with intensity modulated based on depth.

---

## **Processing Steps** ⚙️

### 1. **Optical Flow Estimation** 🌊

- **Methods Used:** 
  - [FastFlowNet](https://github.com/ltkong218/FastFlowNet) - A lightweight network for fast and accurate optical flow estimation
  - Farneback method as a classical alternative

- **FastFlowNet Overview:**
  FastFlowNet is designed for efficient optical flow prediction with several key innovations:
  1. Head Enhanced Pooling Pyramid (HEPP) feature extractor
  2. Center Dense Dilated Correlation (CDDC) layer
  3. Shuffle Block Decoder (SBD)

  ![FastFlowNet Architecture](https://github.com/user-attachments/assets/50dabdaa-222e-4a76-b336-907029ee4f7b)

- **Implementation:**
    - Each pair of consecutive frames is processed to estimate optical flow vectors
    - Flow magnitude and direction computed to identify regions with significant motion
    - Users can choose between methods based on speed/accuracy requirements:
      - FastFlowNet: Better accuracy, GPU-accelerated (~0.1s/frame)
      - Farneback: CPU-friendly, faster but less accurate (~0.05s/frame)

### 2. **Brightening High-Flow Regions** ✨

- **Approach:**
    - Regions with optical flow magnitude above a certain threshold are identified as high-flow areas.
    - These regions are brightened by scaling their pixel intensity based on the flow magnitude.

### 3. **Depth-Guided Brightening** 📏

- **Integration with Depth Maps:**
    - Each frame's corresponding depth map is used to determine the proximity of moving objects.
    - The brightening factor is inversely proportional to the depth value; closer objects receive higher brightness enhancement.
- **Implementation:**
    - The depth maps are normalized and inverted to ensure that closer objects have higher brightness scaling factors.
    - The brightened regions are modulated by the depth-based scaling to achieve the desired effect.

### 4. **Exporting the Modified Video** 🎥

- **Process:**
    - Modified frames with the applied visual effects are compiled back into a video using FFMPEG.
    - The output video maintains the original frame rate and resolution, ensuring synchronization and quality.

---

## **Performance Metrics** 📊

![Performance Metrics](https://github.com/user-attachments/assets/3c7cf95f-edc5-4495-9736-449dcbef6357)

The above visualization shows the performance characteristics across different processing stages and configurations.

- **Optical Flow Estimation:**
    - **FastFlowNet:** 
      - Time per Frame: ~0.1 seconds on NVIDIA GPU
      - Provides good balance of accuracy and speed
    - **Farneback:**
      - Time per Frame: ~0.05 seconds
      - Faster but less accurate than FastFlowNet

- **Brightening and Depth Integration:**
    - **Time per Frame:** Approximately 0.05 seconds.
    - **Efficiency:** Utilized vectorized operations with NumPy and optimized OpenCV functions to ensure real-time processing capabilities.

- **Video Compilation:**
    - **Time:** Dependent on the number of frames; averaged at 0.01 seconds per frame for encoding.
    - **Quality:** Ensured lossless or high-quality compression settings to preserve visual fidelity.

> **Note:** Actual performance may vary based on hardware specifications and input video characteristics.

---

## **Trade-offs, Assumptions, and Limitations** ⚖️

### **Trade-offs**

- **Model Selection vs. Speed:** 
  - FastFlowNet offers excellent accuracy while maintaining computational efficiency
  - Farneback method available as a CPU-friendly alternative for resource-constrained environments
  
- **Depth Map Accuracy:** Assumed that provided depth maps are accurately aligned and correspond to the video frames. Any misalignment can degrade the visual effect's quality.

### **Assumptions**

- **Consistent Frame Rate:** Assumed that the input video has a consistent frame rate matching the depth maps' frame rate.
  
- **Depth Map Alignment:** Depth maps are assumed to be perfectly synchronized and spatially aligned with the video frames.

### **Limitations**

- **Computational Resources:** 
  - FastFlowNet performs best with GPU acceleration
  - Farneback method provides a CPU-friendly alternative with slightly lower accuracy
  - Processing times vary significantly based on chosen method and available hardware

- **Static Background Assumption:** The `PersonSegmenter` module currently uses a simple background subtractor, which may not perform well in dynamic environments or with multiple moving objects.

- **Depth Range Constraint:** Depth maps are constrained to a 0-20 meters range. Objects outside this range may not be accurately represented in the brightness modulation.

- **Optical Flow Flicker:** Initial implementation may exhibit flicker in high-motion regions due to frame-by-frame processing without temporal smoothing.

---

## **Optional Extensions** 🔧

1. **Reduce Optical Flow Flicker**

    - **Implemented:** Applied temporal smoothing using a sliding window of consecutive frames to stabilize optical flow estimations, significantly reducing flicker in the output video.
  
2. **Estimate Scene Flow**

    - **Implemented:** Extended the pipeline to estimate scene flow by combining optical flow with depth information, allowing for more precise identification of moving objects in 3D space.

3. **Beautification**

    - **Implemented:** Applied Gaussian blurring and morphological operations to the brightened regions to create a more aesthetically pleasing visual effect, avoiding harsh transitions.

4. **Comparison of Flow Methods**

    - **Implemented:** Compared FastFlowNet with the classical Farneback method, evaluating differences in:
      - Processing speed (FastFlowNet ~0.1s/frame vs Farneback ~0.05s/frame)
      - Accuracy and flow quality
      - Resource utilization
      - Use cases for each method

---

## **Contact** 📫

For any questions, feedback, or further discussions, please reach out:

- **Ricardo Alexander Martinez**  
  Phone: +1 832.792.9265  
  Email: [martricardo.a@gmail.com](mailto:martricardo.a@gmail.com)

---

## **License** 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements** 🙏

- **FastFlowNet:** [FastFlowNet: A Lightweight Network for Fast Optical Flow Estimation](https://github.com/ltkong218/FastFlowNet)
- **OpenCV:** [Open Source Computer Vision Library](https://opencv.org/)
- **NumPy:** [NumPy](https://numpy.org/)
- **FFMPEG:** [FFMPEG](https://ffmpeg.org/)
- **TQDM:** [TQDM](https://github.com/tqdm/tqdm)

---

## **Appendix** 📚

### **Sample FFMPEG Commands**

To split the video into frames:

```bash
ffmpeg -i input_video.mp4 -vf "fps=60" -start_number 0 data/frames/frame_%07d.png
```

To combine frames into a video:

```bash
ffmpeg -framerate 60 -i data/frames/frame_%07d.png -c:v libx264 -pix_fmt yuv420p outputs/output_visualization.mp4
```

---

## **Running Tests**

Execute unit tests to ensure all modules function as expected:

```bash
python -m unittest discover -s tests
```

---
