# Face Recognition Pipeline

## Used datasets

| Name             | Description                              | Link |
|------------------|------------------------------------------|------|
| celebA_train_500 | The dataset is used to train network     |      |
| celebA_ir        | The dataset is used to calculate TPR@FPR |      |

## Architecture



| Network                                       | Test Accuracy | TPR@FPR (fpr=0.1) | TPR@FPR (fpr=0.1) | TPR@FPR (fpr=0.1) |
|-----------------------------------------------|---------------|-------------------|-------------------|-------------------|
| ResNet18 + Standard Cross-entropy Loss        | 0.75          |                   |                   |                   | 
| ResNet18 + ArcFace + Cross-entropy Loss       | 0.69          |                   |                   |                   |


## References

[1] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou ArcFace: Additive Angular Margin Loss for Deep
Face Recognition https://arxiv.org/pdf/1801.07698.pdf