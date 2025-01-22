# Abstract
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

Code will be released soon ...
