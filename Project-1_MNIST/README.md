## Self-Constructed Neural Networks

### Introduction

In this problem we will investigate handwritten digit classification. MNIST (Modified National Institute of Standards and Technology database) is a large database of handwritten digits commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by “re-mixing” the samples from NIST’s original datasets. The dataset contains 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel grayscale image and is labeled with the correct digit(0-9) it represents. You need to implement one or more neural network to recognize the handwritten digit, and conduct experiment to test your model and conclude the ability of your model. After that, you may implement several modifications to your model and test whether the model acts better.

### Setup

To get started, follow these steps:

1. Clone the GitHub Repository

   ```python
   git clone https://github.com/leo-xfm/DATA130011-nndl-project.git
   ```

2. Set Up Python Environment: Ensure you have a version 3.8 with the packages listed below.

   ```
   pickle
   numpy
   matplotlib.pyplot
   ```

3. Download the dataset with the following command:

   ```bash
   cd dataset
   bash download_mnist.sh
   ```

### Replicating the paper's results

+ Check `quickstart.ipynb` to conduct self-constructed neural networks training and evaluation directly!
+ Run python files in `./scripts` to replicate the results in Table 1 in the reports.

### Get saved models

+ You can get files in `./saved models` from Quark Netdisk.


### Performance

+ We set epochs = 500, batch size = 8192 and patience = 20. It gets the best validation accuracy on epoch 106.

| **Model** | **Val Accuracy** | **Test Accuracy** |
| --------- | ---------------- | ----------------- |
| MLP       | 98.49%           | 98.46%            |

+ Here are the details: 

| **Algorithms**              | **Parameters**                                               |
| --------------------------- | ------------------------------------------------------------ |
| nn.module.MLP               | [784, 512, 128, 10], 'ReLU',  lambda_list=[5e-4]*3 , dropout_rate=None, batch_norm=False |
| nn.optimizer. MomentGD      | init_lr=0.08, mu=0.95                                        |
| nn.lr_scheduler.MultiStepLR | milestones=[50, 100], gamma=0.5                              |
| nn.F.loss_fn                | CrossEntropyLoss                                             |
| tricks                      | Data Augmentation (rotation, translate, brightness, zoom)    |

