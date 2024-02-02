# Face Recognition Pipeline

## Used datasets

| Name             | Description                                                 | Link |
|------------------|-------------------------------------------------------------|------|
| celebA_train_500 | The dataset is used to train network                        |      |
| celebA_ir        | The dataset is used to calculate identification rate metric |      |

## Architecture

### Face Recognition

| Network                                       | Test Accuracy | TPR@FPR (fpr=0.1) | TPR@FPR (fpr=0.1) | TPR@FPR (fpr=0.1) |
|-----------------------------------------------|---------------|-------------------|-------------------|-------------------|
| ResNet18 + Standard Cross-entropy Loss        |               |                   |                   |                   | 
| ResNet18 + ArcFace + Cross-entropy Loss       |               |                   |                   |                   |

