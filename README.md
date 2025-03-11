# LiteNeXt
Official code repository for: LiteNeXt - A Novel Lightweight ConvMixer-based Model with Self-embedding
Representation Parallel for Medical Image Segmentation

Links to the paper:
+ [Biomedical Signal Processing and Control](https://www.sciencedirect.com/science/article/pii/S1746809425002848?via%3Dihub)
+ Full Code in progress ...

# 1. Abstract
The emergence of deep learning techniques has advanced the image segmentation task, especially
for medical images. Many neural network models have been introduced in the last decade bringing the
automated segmentation accuracy close to manual segmentation. However, cutting-edge models like
Transformer-based architectures rely on large scale annotated training data, and are generally designed
with densely consecutive layers in the encoder, decoder, and skip connections resulting in large number
of parameters. Additionally, for better performance, they often be pretrained on a larger data, thus
requiring large memory size and increasing resource expenses. In this study, we propose a new
lightweight but efficient model, namely LiteNeXt, based on convolutions and mixing modules with
simplified decoder, for medical image segmentation. The model is trained from scratch with small
amount of parameters (0.71M) and Giga Floating Point Operations Per Second (0.42). To handle
boundary fuzzy as well as occlusion or clutter in objects especially in medical image regions, we propose
the Marginal Weight Loss that can help effectively determine the marginal boundary between object and
background. Furthermore, we propose the Self-embedding Representation Parallel technique, that can
help augment the data in a self-learning manner. Experiments on public datasets including Data Science
Bowls, GlaS, ISIC2018, PH2, and Sunnybrook data show promising results compared to other state-ofthe-art CNN-based and Transformer-based architectures.

# 2. Architecture


<p align="center">
	<img , src="https://github.com/user-attachments/assets/8f4eb3d2-951d-41ca-8fdc-5c6567e3edd3"> <br />
	<em>
		Figure 1: General architecture of the proposed LiteNeXt model. (a) Overview of the training pipeline,
(b) Overview of the inference phase, (c) Architecture of the CLG block, (d) Architecture of the Residual
Block-abbreviated as RB block, (e) Architecture of the Head Prediction
	</em>
</p>

<p align="center">
	<img , src="https://github.com/user-attachments/assets/4341d494-cc0a-46d5-acb2-64d1b0459360"> <br />
	<em>
		Figure 2: Overview architectures of (a) Our Proposed LGEMixer block, (b) LocalMixer block,
(c) FarMixer block
	</em>
</p>

<p align="center">
	<img , src="https://github.com/user-attachments/assets/bad65495-5468-4252-b3ad-8f7652f6deb6"> <br />
	<em>
		Figure 3: Overview architectures of Head Projector, (a) Architecture of ProjectorT, Architecture of
ProjectorS
	</em>
</p>

<p align="center">
	<img , src="https://github.com/user-attachments/assets/b7b8d727-4483-4666-a55e-8a53aff3e457"> <br />
	<em>
		Figure 3: Overview of the weighting process for each object partition, wb is the background weight, wo
is the object weight, and wm is the marginal weight
	</em>
</p>

# 3. Reference
```
@article{tran2024litenext,
  title={LiteNeXt: A Novel Lightweight ConvMixer-based Model with Self-embedding Representation Parallel for Medical Image Segmentation},
  author={Tran, Ngoc-Du and Tran, Thi-Thao and Nguyen, Quang-Huy and Vu, Manh-Hung and Pham, Van-Truong},
  journal={arXiv preprint arXiv:2405.15779},
  year={2024}
}
```


