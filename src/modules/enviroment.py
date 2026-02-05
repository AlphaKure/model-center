from src.schemas.enviroment import TorchInformation

import torch


def torch_test() -> TorchInformation:

    """test torch available"""
    return TorchInformation(
        version= str(torch.__version__), # torch version
        is_available= torch.cuda.is_available(),
        device_count= torch.cuda.device_count(),
        devices = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    )
