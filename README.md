# Face Recognition Pipeline

## Used datasets

| Name             | Description                              | Link                                      |
|------------------|------------------------------------------|-------------------------------------------|
| celebA_train_500 | The dataset is used to train network     | https://disk.yandex.ru/d/S8f03spLIA1wrw   |
| celebA_ir        | The dataset is used to calculate TPR@FPR | https://disk.yandex.com/d/KN4EEkNKrF_ZXQ  |

## Architecture

| Network                                       | Test Accuracy | TPR@FPR (fpr=0.05)     | TPR@FPR (fpr=0.1)      | TPR@FPR (fpr=0.2)      | TPR@FPR (fpr=0.2)      |
|-----------------------------------------------|---------------|------------------------|------------------------|------------------------|------------------------|
| ResNet18 + Standard Cross-entropy Loss        | 0.75          | thr = 0.69, tpr = 0.65 | thr = 0.67, tpr = 0.76 | thr = 0.63, tpr = 0.87 | thr = 0.57, tpr = 0.97 | 
| ResNet18 + ArcFace + Cross-entropy Loss       | 0.69          |                        |                        |                        |                        |


## References

[1] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou ArcFace: Additive Angular Margin Loss for Deep
Face Recognition [link](https://arxiv.org/pdf/1801.07698.pdf)

[2] Jiaheng Liu*1, Haoyu Qin*2, Yichao Wu 2, Ding Liang AnchorFace: Boosting TAR@FAR for Practical Face Recognition [link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi1gpyMp_6DAxVsBxAIHb_dDb4QFnoECA8QAQ&url=https%3A%2F%2Fojs.aaai.org%2Findex.php%2FAAAI%2Farticle%2Fview%2F20063%2F19822&usg=AOvVaw13OueGt-qm3zfYg7XIlOtg&opi=89978449)