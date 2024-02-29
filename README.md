Project of Brain decoding through AI

Brain decoding is the task of classify Medical imagery taken in the execution of specific cognitive task.
Usually accomplish through complicated hand-crafted feature selection combined with conventional Machine Learning Technics.
A first Milestone for the usage of AI for this purpose can be found in the work of Wang et Al. 
Relying on the HCP Dataset, briefly, it contains fMRI of a thousand of subject executing seven different cognitive tasks supposedly representative of cognition.

In this work, the purpose is to transfer a model pre-trained on the HCP Dataset to gather insight on other dataset detained by the LNPC institute in Grenoble, France.
Those datasets contain fMRI data on subtask involved in language processing with few participants (10 to 20). Such as semantic, monitoring, production, phonetic ...

To accomplish such purpose we reapplied the work made by Wang,

- First we selected a specific range of models pretrained on HCP, we also pretrained some on the HCP dataset using different normalisation preprocessing regarding the 4D nature of the data
- Second we train those models on the languages dataset
- Third, we use the Guided Backpropagation Method to gather knowledge on the specific areas of the brain involved in the cognitive processing.
- Forth, SVM, are used on the same dataset to compare performances with no preprocessing.

The scripts are therefore articulated around 3 main tasks :
- Preprocessing the data (For the machine learning purpose as the data already fulfill the main need for fMRI analysis)
- Training the models
- Evaluating their performances
- Analysis of the model choice (GBP)




Conclusion:
  The result of such project were inclined to confirm the effectiveness of using pretrained model, even in a different Domain/task setup.
  Especially when comparing to SVM.
  However, they still lacked stability in the training process, which was highlighted through the guided backpropagation method
