# Diffusion Model for ARCorpus

[**DiMARCo**] Diffusion Model for ARCorpus

- Stay away from the hottest trending of LLM that being considered as AGI and being able to solve anything in the world;
- I still see ARC as a **visual task** and should be solved with **vision model** and **reasoning model**.

--------------------------
## To-Do List

- [x] **Diffusion Model** for all tasks (as foundation model)
- [x] **LoRA** Training for single task
- [ ] Random / Grid / Neural **ARChitecture Search** for LoRA-training optimization
- [ ] **Optical-Flow** as ControlNet Module
- [ ] **Video-Interpolation** as Reasoning Module

--------------------------
## Future Improvements

--------------------------
## Dataset
- [ARC Competition 2024](https://www.kaggle.com/competitions/arc-prize-2024/data)
- [ARCkit](https://github.com/mxbi/arckit/tree/main/arckit/data)

--------------------------
## Getting started
This code was tested with:

* Windows 11
* NVIDIA GeForce RTX 3060 - 6144 MiB
* Python 3.10.6
* CUDA 12.1

Setup environment:
```shell
pip install -r requirements.txt
```

Install PyTorch depending on **CUDA version**:
```shell
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

Install special libraries:
```shell
pip install pipwin
pipwin install cairocffi
!pipwin install cairosvg
```

#### Download **Checkpoints**

--------------------------
## User Guide

#### Run
```shell
python -m src.main
```

--------------------------
## References

- **ARC Solvers**
    - [DreamCoder](https://github.com/mxbi/dreamcoder-arc)
    - [ARC-RevEng](https://github.com/michaelhodel/re-arc)
    - [ARC-DSL](https://github.com/michaelhodel/arc-dsl)
    - [Graph-Abstractions](https://github.com/khalil-research/ARGA-AAAI23)

- **Optical Flow**
    - [2D-RAFT](https://github.com/princeton-vl/RAFT)
    - [3D-RAFT](https://github.com/princeton-vl/RAFT-3D)
    - [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)

- **Diffusion Model**
    - [🤗 Course: Diffusion](https://huggingface.co/learn/diffusion-course)
    - [Denoising-Diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch)

- **Spatial Feature & Correlation**
    - [Spatial-Correlation](https://github.com/ClementPinard/Pytorch-Correlation-extension)
    - [Dilated-Convolution-with-Learnable-Spacings](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch)