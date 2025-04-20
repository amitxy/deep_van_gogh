# Deep Van Gogh - DL Project - Van Gogh & Style Transfer
This project explores the intersection of deep learning and art, focusing on classifying and generating paintings in the style of Vincent Van Gogh. It has been a major part of a course we took - "Introduction to Deep Learning" at TAU's faculty of engineering in 2024-2025, during the 3rd year of our studies.

The project consists of 2 main parts:

**Part 1 - Transfer Learning & Fine Tuning:**

1st, we used 2 pre-trained Convolutional Neural Networks (CNNs): AlexNet and VGG-19, and fine-tuned them on a dataset of post-impressionist paintings with a particular emphasis on Van Gogh’s work.
Our goal was to build an accurate classifier capable of distinguishing between Van Gogh and other post-impressionist artists, while also learning meaningful visual features.
We used key techniques and tools, such as:
- *Transfer Learning and Feature Extraction*
- *Hyperparameter Tuning with [Optuna](https://optuna.org/)*
- *Experiment Tracking with [Weights & Biases (W & B)](https://wandb.ai/) API*
- *K-Fold Cross Validation*
- *Regularization and Data Augmentation techniques*

**Part 2 - Style Transfer:**

In this part, we implemented [Style Transfer](https://arxiv.org/abs/1508.06576) to generate new images that blend the content of our own personal images with the style of Van Gogh’s paintings.
By doing so, we created stylized artworks that visually resemble Van Gogh’s style while preserving the structure of the original content images.

We then used the fine-tuned CNN models from Part 1 to test how well they could classify these newly generated images as Van Gogh, testing whether the transferred style was strong enough to be recognized as a Van Gogh painting. 

Throughout the project, we explored how deep models understand artistic styles (styles? or explicitly Van Gogh styles?) and how well they generalize. We continuously compared the performance of AlexNet and VGG-19 - both in terms of classification accuracy and their ability to support artistic generation, while gaining hands-on experience with CNN fine-tuning, model optimization, and experiment tracking using tools like Optuna and W&B.

The project's final grade - 100.
