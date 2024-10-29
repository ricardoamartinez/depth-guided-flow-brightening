import tensorrt as trt
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
import pycuda.driver as cuda
import pycuda.autoinit

console = Console()

class TRTInference:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.allocate_buffers()

    def allocate_buffers(self):
        """Allocate CUDA memory for inputs/outputs"""
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img1, img2):
        """Run inference on two frames"""
        # Preprocess input images
        preprocessed = self.preprocess_images(img1, img2)
        
        # Copy input to device
        cuda.memcpy_htod(self.inputs[0]['device'], preprocessed)
        
        # Run inference
        self.context.execute_v2(
            bindings=[inp['device'] for inp in self.inputs] + 
                     [out['device'] for out in self.outputs]
        )
        
        # Copy output back to host
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # Post-process output
        return self.postprocess_output(self.outputs[0]['host'])

    def preprocess_images(self, img1, img2):
        """Preprocess images for network input"""
        # Convert to float32 and normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Stack images
        stacked = np.stack([img1, img2], axis=0)
        
        # Add batch dimension if needed
        if len(stacked.shape) == 3:
            stacked = np.expand_dims(stacked, axis=0)
            
        return stacked.ravel()

    def postprocess_output(self, output):
        """Post-process network output"""
        # Reshape output to original dimensions
        flow = output.reshape(1, 2, self.height, self.width)
        return flow[0].transpose(1, 2, 0)

class FastFlowNetTRT:
    def __init__(self):
        """Initialize FastFlowNet with TensorRT support"""
        self.engine_path = Path(__file__).parent.parent / 'models' / 'engines' / 'fastflownet.engine'
        self.trt_inference = None
        
        if not self.engine_path.exists():
            console.print("[yellow]TensorRT engine not found. Please build it first.[/yellow]")
            console.print("Run the following commands in the FastFlowNet docker container:")
            console.print("[cyan]cd /usr/share/dev/FastFlowNet[/cyan]")
            console.print("[cyan]python ./tensorrt_workspace/fastflownet.py[/cyan]")
            return
            
        try:
            self.trt_inference = TRTInference(str(self.engine_path))
            console.print("[green]Successfully loaded TensorRT engine[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load TensorRT engine: {e}[/red]")

    def __call__(self, img1, img2):
        """Run inference on two frames"""
        if self.trt_inference is None:
            raise RuntimeError("TensorRT engine not initialized")
            
        return self.trt_inference.infer(img1, img2) 