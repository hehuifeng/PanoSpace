# Unlocking single-cell level and whole-tissue insights in spatial transcriptomics with PanoSpace

---

## ⚠️ Important Notice

> **This repository contains the implementation used in the paper:**  
> **He, H.F., Peng, P., Yang, S.T. et al.**  
> *Unlocking single-cell level and continuous whole-slide insights in spatial transcriptomics with PanoSpace.*  
> **Nature Computational Science (2026)**  
> DOI: https://doi.org/10.1038/s43588-025-00938-y

> **The actively developed and redesigned version of PanoSpace is available here:**  
> 👉 **https://github.com/hehuifeng/PanoSpace-core**
---





![PanoSpace_overview](figures/fig1.png)

**PanoSpace** is a Python package designed for single-cell level analysis and visualization of low-resolution spatial transcriptomics data, such as Visium. By integrating scRNA-seq data, low-resolution spatial transcriptomics data, and high-definition H&E-stained images from the same tissue region or type, PanoSpace transforms spot-level spatial transcriptomics data into detailed, whole-tissue single-cell insights. This enables accurate cell detection, cell type annotation, and gene expression prediction at a single-cell resolution. Additionally, PanoSpace facilitates in-depth analysis of intra-cell type heterogeneity and cell-cell interactions, unlocking new dimensions of spatial biology.

This repository is currently under construction.

### Installation

**PanoSpace** has been developed and tested on **Ubuntu 22.04.4 LTS (GNU/Linux 6.5.0-18-generic x86_64)**.

To install **PanoSpace**, follow these steps:

1. **Set up a Conda environment for PanoSpace**:

    ```bash
    conda env create -f environment.yml
    conda activate PanoSpace
    pip install torch==2.4.0 torchvision==0.19.0 pytorch-lightning==2.1.2
    ```

    In this step, we install PyTorch version `2.4.0` with CUDA `12.2` support.

    > **Note:** The `gurobipy` dependency specified in `environment.yml` is commercial software. However, students and staff members of academic institutions are eligible for a free, full-featured license. For more details, visit https://pypi.org/project/gurobipy/

2. **Clone the repository and install PanoSpace**:

    ```bash
    git clone https://github.com/hehuifeng/PanoSpace.git
    cd PanoSpace
    pip install .
    ```

3. **Verify the installation in Python**:

    ```python
    import panospace as ps
    print(ps.__version__)
    # Output: '0.1.0'
    ```

    Installation time may vary depending on your network environment and hardware configuration but typically completes within a few minutes.

## Reproducibility
We provide source codes for reproducing the PanoSpace analysis in the main text in the `demos` directory.

- [10x Visium Breast Cancer dataset](demo/Visium_Breast_Reproducibility.ipynb)
- [10x Visium adult mouse olfactory bulb dataset](demo/Visium_bulb_Reproducibility.ipynb)
 
## Contact information
Please contact Hui-Feng He (<huifeng@mails.ccnu.edu.cn>) or Prof. Xiao-Fei Zhang (<zhangxf@ccnu.edu.cn>) if any enquiry.
