
<div align="center">

<samp>

<h2> Two-stage Contextual Transformer-based Convolutional Neural Network for Airway Extraction from CT Images </h1>

<h4> Yanan Wu, Shuiqing Zhao, Shouliang Qi*, Jie Feng, Haowen Pang, Runsheng Chang, Long Bai, Mengqi Li, Shuyue Xia, Wei Qian, Hongliang Ren* </h3>

</samp>   

</div>     
    
---

If you find our code or paper useful, please cite as (This paper in journal version will be updated later)

```bibtex
@article{wu2022two,
  title={Two-stage Contextual Transformer-based Convolutional Neural Network for Airway Extraction from CT Images},
  author={Wu, Yanan and Zhao, Shuiqing and Qi, Shouliang and Feng, Jie and Pang, Haowen and Chang, Runsheng and Bai, Long and Li, Mengqi and Xia, Shuyue and Qian, Wei and others},
  journal={arXiv preprint arXiv:2212.07651},
  year={2022}
}
```

---
## Abstract
Accurate airway segmentation from computed tomography (CT) images is critical for planning navigation bronchoscopy and realizing a quantitative assessment of airway-related chronic obstructive pulmonary disease (COPD). Existing methods face difficulty in airway segmentation, particularly for the small branches of the airway. These difficulties arise due to the constraints of limited labeling and failure to meet clinical use requirements in COPD. We propose a two-stage framework with a novel 3D contextual transformer for segmenting the overall airway and small airway branches using CT images. The method consists of two training stages sharing the same modified 3D U-Net network. The novel 3D contextual transformer block is integrated into both the encoder and decoder path of the network to effectively capture contextual and long-range information. In the first training stage, the proposed network segments the overall airway with the overall airway mask. To improve the performance of the segmentation result, we generate the intrapulmonary airway branch label, and train the network to focus on producing small airway branches in second training stage. Extensive experiments were performed on in-house and multiple public datasets. Quantitative and qualitative analyses demonstrate that our proposed method extracts significantly more branches and longer lengths of the airway tree, while accomplishing state-of-the-art airway segmentation performance.  

<p align="center">
<img src="abstract.png" alt="Transformerairway" width="1000"/>
</p>


---
## Environment

- SimpleITK
- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- tqdm
- pickle

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 
 
- CoT-Unet.py : 3D CoT module.

---
## Dataset
1. The in-house dataset was provided by the hospital. 
2. The ISICDM dataset was released in the ISICDM 2021 challenge and labeled by the challenge organizer.
3. [EXACT’09 challenge dataset](http://image.diku.dk/exact/)
4. Binary airway segmentation (BAS) dataset: This can be found in the paper [Learning tubule-sensitive CNNs for pulmonary airway and artery-vein segmentation in CT](https://arxiv.org/abs/2012.05767) 
5. [Airway Tree Modeling 2022 (ATM22) dataset](https://atm22.grand-challenge.org/)
---



## References
Code adopted and modified from:
1. CoTNet model
    - Paper [Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).
    - official PyTorch implementation [Code](https://github.com/JDAI-CV/CoTNet.git).
---

## Other information
MICCAI Challenge ATM2022 4th place, More details will be added soon.

## Contact
For any queries, please raise an issue or contact [Yanan Wu](mailto:yananwu513@gmail.com) and [Shuiqing Zhao](ddddddd).

---
