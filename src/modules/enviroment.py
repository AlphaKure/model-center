from src.schemas.enviroment import TorchInfomation

import torch


def torch_test() -> TorchInfomation:

    """test torch available"""
    return TorchInfomation(
        version= str(torch.__version__), # torch version
        isAvailable= torch.cuda.is_available(),
        deviceCount= torch.cuda.device_count(),
        devices = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    )
