# UFZ-data-fusion
This repository is the code of "An enhanced day-night feature fusion method for fine-grained urban functional zone mapping from the SDGSAT-1 imagery".
# Dependency
1. Tensorflow
2. Keras
3. numpy
4. os
5. scipy
# Essential function
fusion.py

![figure_framework](https://github.com/user-attachments/assets/a4c745dd-d29d-44ad-90d2-b031090ea824)
The model is in file "fusion.py". There are three essential classes in the file, the CFA module, the multi-level attention aggregation module and the context-enhanced fusion module. This three classes is corresponding to three parts in our paper.

train.py
train.py is a training example. You can configure your dataset for training.
