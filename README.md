# LiteNeXt
Official code repository for: LiteNeXt - A Novel Lightweight ConvMixer-based Model with Self-embedding
Representation Parallel for Medical Image Segmentation (**Preprint submitted to Elsevier**)

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
	<img width=900, src="https://github.com/user-attachments/assets/9678daeb-72be-48e9-bcd7-485ca3ab793a"> <br />
	<em>
		Figure 1: Illustration of the proposed FCBFormer architecture
	</em>
</p>

# 3. Results 
![image](https://github.com/user-attachments/assets/fb149108-9d0a-4a2e-bfaf-68f70f601772)
![image](https://github.com/user-attachments/assets/708553a6-1c03-424a-ab45-58a96e944d57)
![image](https://github.com/user-attachments/assets/a526a3d5-28c3-4fcd-a1b9-dab2a2069e35)
![image](https://github.com/user-attachments/assets/04327b13-4b92-4712-8923-222f317ef0d6)




