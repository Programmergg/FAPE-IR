<p align="center">
  <img src="figs/logo.png" height="240">
</p>

### FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration [CVPR 2026 Accept]
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Programmergg.FAPE-IR)
[![GitHub Stars](https://img.shields.io/github/stars/Programmergg/FAPE-IR?style=social)](https://github.com/Programmergg/FAPE-IR)

> [[Paper (arXiv)](https://arxiv.org/abs/2511.14099)] &emsp;
> **Jingren Liu**<sup>&#42;</sup>, **Shuning Xu**<sup>&#42;</sup>, **Qirui Yang**<sup>&#42;</sup>, **Yun Wang**, **Xiangyu Chen**<sup>✉</sup>, **Zhong Ji**<sup>✉</sup> <br>
> <small><sup>&#42;</sup>Equal contribution &emsp; <sup>✉</sup>Corresponding author</small>


---
*This repo is an early exploration of unified image restoration (unified understanding & generation). We are actively investigating more lightweight designs and alternatives beyond the MLLM + Diffusion paradigm, and will continuously maintain and update this repo with new progress.*

*The three authors are struggling with their doctoral dissertations. Please wait a moment while they work on cleaning the code and providing new directions and suggestions.*
---

## Status

🚧 **Coming Soon (Open-sourcing in progress).**

We are preparing:
- clean and reproducible code (training / inference / evaluation)
- pretrained checkpoints
- documentation and scripts

Please **star** this repo to get updates.

---

## 🚩 New Features/Updates
- ✅ Nov 25, 2025. Release the arXiv paper.
- 🚧 TBD. Release training code.
- 🚧 TBD. Release pretrained checkpoints & model zoo.
- 🚧 TBD. Release evaluation scripts and example results.

---

## :book: Resources

### Checkpoints / Datasets / Results (TBD)

|Item|Link|
|:-|:-|
|Pretrained checkpoints|TBD|
|Testset (GT/LQ)|TBD|
|Visual results|TBD|
|Compared methods|TBD|

---

## :computer: Usage (TBD)

### :arrow_right: Environment
```bash
conda create -n fapeir python=3.11 -y
conda activate fapeir
pip install -r requirements.txt
```

### :arrow_right: Inference

**Single image**

```bash
python inference.py --input ./examples/0001.png --output ./results
```

**Folder**

```bash
python inference.py --input ./examples --output ./results
```

### :arrow_right: Evaluation

```bash
python eval.py --inp_imgs ./results --gt_imgs ./dataset/GT --save_dir ./logs
```

### :arrow_right: Training

```bash
bash train.sh
```

---

## :bar_chart: Results

### Quantitative Results

<p align="center">
  <img src="./figs/table1.png" width="800">
  <br>
  <em>Figure 1. Quantitative comparison with state-of-the-art AIO-IR methods on six tasks (deraining, denoising, deblurring, desnowing, dehazing, low-light enhancement). <b>Best</b> in red, <u>second-best</u> in blue.</em>
</p>

<p align="center">
  <img src="./figs/table2.png" width="400">
  <br>
  <em>Figure 2. Unified comparison across SR task series. FAPE-IR achieves the best PSNR / SSIM / LPIPS / FID / DISTS on the majority of metrics.</em>
</p>

### Qualitative Results

<p align="center">
  <img src="./figs/supp1.png" width="900">
  <br>
  <em>Figure 6. Qualitative comparison among unified models, including BAGEL, Nexus-Gen, Uniworld-V1, and Emu3.5.</em>
</p>

<p align="center">
  <img src="./figs/supp2.png" width="900">
  <br>
  <em>Figure 7. Qualitative comparison of restoration results produced by FAPE-IR and state-of-the-art AIO-IR models.</em>
</p>

---

## Citation

If you use this work, please cite:

```bibtex
@article{liu2025fape,
  title={FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration},
  author={Liu, Jingren and Xu, Shuning and Yang, Qirui and Wang, Yun and Chen, Xiangyu and Ji, Zhong},
  journal={arXiv preprint arXiv:2511.14099},
  year={2025}
}
```

---

## Acknowledgement

We thank all collaborators and colleagues for their helpful discussions and support. We especially thank Dr. Xiangyu Chen and Professor Zhong Ji for their guidance and revisions to this work.

---

## Contact

If you have any questions, feel free to reach out:

* Email: jrl0219@tju.edu.cn
