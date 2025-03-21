Description
-----------
The implementation of the AAAI 2025 paper "Densest k-Subgraph Mining via a Provably Tight Relaxation" and its extended version ["On Densest k-Subgraph Mining and Diagonal Loading"](https://arxiv.org/pdf/2410.07388). This demo applies the proposed algorithms to the Facebook dataset with various values of k.

The repository also includes an implementation of L-ADMM, as proposed in the WSDM 2021 paper ["Exploring the Subgraph Density-Size Trade-off via the Lovaśz Extension"](https://dl.acm.org/doi/abs/10.1145/3437963.3441756) by Aritra Konar and Nicholas D. Sidiropoulos.

Instruction
-----------

Operating system: Ubuntu 22.04.3 LTS

Python version: 3.10.12

Install dependencies: pip3 install -r requirements.txt

Run the demo: python3 demo.py

Citation
-----------

@article{lu2024densest,
  title={On Densest $k$-Subgraph Mining and Diagonal Loading},
  author={Lu, Qiheng and Sidiropoulos, Nicholas D and Konar, Aritra},
  journal={arXiv preprint arXiv:2410.07388},
  year={2024}
}

@inproceedings{lu2025densest,
title={Densest $k$-Subgraph Mining via a Provably Tight Relaxation},
author={Lu, Qiheng and Sidiropoulos, Nicholas D and Konar, Aritra},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
year={2025}
}
