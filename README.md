# Project of Brain decoding through AI

## Description
Brain decoding is the task of classify Medical imagery taken in the execution of specific cognitive task.
Usually accomplish through complicated hand-crafted feature selection combined with conventional Machine Learning Technics.
A first Milestone for the usage of AI for this purpose can be found in the work of Wang et Al. 
Relying on the HCP Dataset, briefly, it contains fMRI of a thousand of subject executing seven different cognitive tasks supposedly representative of cognition.

In this work, the purpose is to transfer a model pre-trained on the HCP Dataset to gather insight on other dataset detained by the LNPC institute in Grenoble, France.
Those datasets contain fMRI data on subtask involved in language processing with few participants (10 to 20). Such as semantic, monitoring, production, phonetic ...

## Method
To accomplish such purpose we mainly reapplied the work made by Wang et al., (2020)

- First we selected a specific range of models pretrained on HCP, we also pretrained some on the HCP dataset using different normalisation preprocessing regarding the 4D nature of the data
- Second we train those models on the languages dataset
- Third, we use the Guided Backpropagation Method to gather knowledge on the specific areas of the brain involved in the cognitive processing.
- Forth, SVM, are used on the same dataset to compare performances with no preprocessing.

The scripts are therefore articulated around 3 main tasks :
- Preprocessing the data (For the machine learning purpose as the data already fulfill the main need for fMRI analysis)
- Training the models
- Evaluating their performances
- Analysis of the model choice (GBP)



## Results
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$  Phonolgy $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$      Control
![FMRI GBP](https://github.com/BoustieA/Brain-decoding/assets/124263630/b64c1280-3595-4d79-93d2-a4d5f472d62f)

As higlighted in this pictures, the region are indeed involved in the related cognitve tasks.
However, there is no particular fine grained details that could helps our understanding of the language through cooperation of different areas.
We correlated the limitaion of such results to the lack of stability in training our models thus to the few samples to our disposal.

## Conclusion
The result of such project were inclined to confirm the effectiveness of using pretrained model, even in a different Domain/task setup.
Especially when comparing to SVM.
  However, they still lacked stability in the training process, which was highlighted through the guided backpropagation method




## References
Wang, X., Liang, X., Jiang, Z., Nguchu, B. A., Zhou, Y., Wang, Y., Wang, H., Li, Y., Zhu, Y., Wu, F., Gao, J.,
      & Qiu, B. (2020). Decoding and mapping task states of the human brain via deep learning. Human
Brain Mapping, 41(6), 1505–1519. https://doi.org/10.1002/hbm.24891
