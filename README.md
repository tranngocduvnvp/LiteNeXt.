# LiteNeXt
Official code repository for: LiteNeXt - A Novel Lightweight ConvMixer-based Model with Self-embedding
Representation Parallel for Medical Image Segmentation

Links to the paper:
+ [arXiv](https://arxiv.org/pdf/2405.15779)

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
	<img , src="https://github.com/user-attachments/assets/9678daeb-72be-48e9-bcd7-485ca3ab793a"> <br />
	<em>
		Figure 1: General architecture of the proposed LiteNeXt model. (a) Overview of the training pipeline,
(b) Overview of the inference phase, (c) Architecture of the CLG block, (d) Architecture of the Residual
Block-abbreviated as RB block, (e) Architecture of the Head Prediction
	</em>
</p>

<p align="center">
	<img , src="https://github.com/user-attachments/assets/b7b8d727-4483-4666-a55e-8a53aff3e457"> <br />
	<em>
		Figure 2: Overview of the weighting process for each object partition, wb is the background weight, wo
is the object weight, and wm is the marginal weight
	</em>
</p>


# 3. Qualitative results 
<p align="center">
	<img , src="https://github.com/user-attachments/assets/8b384b91-f366-4699-9365-caa84aef81ad"> <br />
	<em>
		Figure 3: Visualization results of the top5 best predictions on the Bowl2018 dataset
	</em>
</p>

<p align="center">
	<img , src="https://github.com/user-attachments/assets/aaaae6f5-2abb-4b61-b761-018c394d8651"> <br />
	<em>
		Figure 4: Visualization results of the top5 best predictions on the GlaS dataset
	</em>
</p>
<p align="center">
	<img , src="https://github.com/user-attachments/assets/f709b01d-8854-4e38-b9d2-0993c743ab79"> <br />
	<em>
		Figure 5: Visualization results of the top5 best predictions on the ISIC2018 dataset
	</em>
</p>
<p align="center">
	<img , src="https://github.com/user-attachments/assets/1e36a5fb-7aa3-4dcf-855c-e17cc854119a"> <br />
	<em>
		Figure 6: Visualization results of the top5 best predictions on the PH2 dataset
	</em>
</p>


