# coding: utf-8

import torch

### Required to recognize Intel Arc GPU (XPU) in device handling
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

### Detect available GPU/CPU device, returns string identifier (and device object for PyTorch operations) 
def device_detect():
    try:
        import torch
        
        if torch.cuda.is_available():
            # NVIDIA GPU
            return "cuda", torch.device('cuda')
        
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel Arc GPU
            return "xpu", torch.device('xpu')

        elif hasattr(torch, 'tpu') and torch.tpu.is_available():
            # Google TPU
            try:
                import torch_xla.core.xla_model as xm
                return "tpu", xm.xla_device()  
            except ImportError:
                pass  # TPU libraries not available

        # Fallback to CPU
        return "cpu", torch.device('cpu')

    except ImportError:
        return "cpu", None  # Torch not available
