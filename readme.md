# Model Center

Local model playground.

## Requirements

- uv
- python 3.10

## Installer

```shell
uv pip install -e ".[cuda]" # For NVIDIA GPUs
uv pip install -e ".[amd]" # For AMD GPUs
```

## Test enviroment

CPU: AMD Threadripper 7995WX

GPU: RTX 5090 32GB*1

RAM: RDIMM DDR5 128GB*2

## Support Models

### Text to Image

* Running well at cuda mode:

    - [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)

    - [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

* Running well at balanced mode

    - [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
