# An Investigation of the Effectiveness of Phase for Audio Classification

- Authors: Shunsuke Hidaka, Kohei Wakamiya, Tokihiko Kaburagi
- Paper: [https://ieeexplore.ieee.org/document/9746037](https://ieeexplore.ieee.org/document/9746037)

## About

This repository contains the implementation of "[An Investigation of the Effectiveness of Phase for Audio Classification](https://ieeexplore.ieee.org/document/9746037)".
In this paper, we proposed LEAF-extended, a learnable front-end that can compute the phase and its derivatives.
Our code is written by using PyTorch.
The original LEAF implementation, which calculates only the amplitude, is available at the following link: [https://github.com/google-research/leaf-audio](https://github.com/google-research/leaf-audio) (TensorFlow implementation).

## Contents

- `notebooks/`: demonstrations, building datasets
  - `complexnn.ipynb`: the complex batch normalization
  - `data.ipynb`: a dataset class
  - `format_datasets.ipynb`: **building datasets**
  - `leaf.ipynb`: LEAF-extended
  - `phase.ipynb`: some phase functions
  - `specaug.ipynb`: SpecAugment
- `scripts/`: Python scripts
- `shells/`: example shell scripts

## Dependencies

See `pyproject.toml`.

```toml
python = "^3.8"
einops = "^0.3.0"
librosa = "^0.8.1"
matplotlib = "^3.4.2"
numpy = "^1.20.3"
pandas = "^1.2.4"
pyarrow = "^4.0.1"
pytorch-lightning = "^1.3.4"
seaborn = "^0.11.1"
SoundFile = "^0.10.3"
PyYAML = "^5.4.1"
torch = "^1.8.1"
```

## Reference

Link: [An Investigation of the Effectiveness of Phase for Audio Classification](https://ieeexplore.ieee.org/document/9746037)

```bibtex
@INPROCEEDINGS{9746037,
  author={Hidaka, Shunsuke and Wakamiya, Kohei and Kaburagi, Tokihiko},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={An Investigation of the Effectiveness of Phase for Audio Classification}, 
  year={2022},
  volume={},
  number={},
  pages={3708-3712},
  doi={10.1109/ICASSP43922.2022.9746037}}
```
