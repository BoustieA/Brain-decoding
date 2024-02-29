import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
from nilearn.image import clean_img
from nilearn.image import mean_img
from nilearn.input_data import NiftiMasker
from nilearn.maskers import MultiNiftiMasker

class Preprocesser:
    def __init__(self,average_time=False,normalisation="global",detrend=False,thresh_PCA=2):
        """
        Preprocess list of scan FMRI
        normalisation : "global"/"time"
        detrend : True/False # remove linear trend of data along time axis, equivalent of removing the linear regression component
        average_time : True/False # wether to average samples over time axis or not
        thresh_PCA : #keep an amount of features so that strictly less than x% 
                     #percent of the total variance is kept according to the SVD algorithm

        """
        self.PCA=PCA
        self.thresh_PCA=2 
        self.n_features_to_keep=None

        
        self.masker=None

        self.avg_time=average_time
        self.detrend=detrend
        self.normalisation=normalisation

    def fit(self,list_imgs):
        if self.avg_time:
            list_imgs=self.average(list_imgs)#average time sequence
        self.get_mask(list_imgs)#compute mask
        list_imgs=self.mask_list_img(list_imgs,self.masker)
        if self.normalisation=="global":#global_norm
            list_imgs=self.standardize(list_imgs)
        self.PCA=self.PCA(n_components=list_imgs.shape[0])#n_sample as it is the min for sklearn PCA
        self.PCA.fit(list_imgs)#fit PCA 
        coef=self.PCA.explained_variance_ratio_#get coeff of importance for total variance
        coef=np.cumsum(coef)
        features_to_keep=np.nonzero(coef<self.thresh_PCA)[0]#get number of features to keep x% of total variance
        self.n_features_to_keep=len(features_to_keep)
        list_imgs=self.PCA.transform(list_imgs)[:,:self.n_features_to_keep]
        return list_imgs

    def transform(self,list_imgs):
        if self.avg_time:
            list_imgs=self.average(list_imgs)
        list_imgs=self.mask_list_img(list_imgs,self.masker)
        if self.normalisation=="global":
            list_imgs=self.standardize(list_imgs)
        list_imgs=self.PCA.transform(list_imgs)[:,:self.n_features_to_keep]
        return list_imgs

    def average(self,list_img):
        print(list_img[0].shape)
        L=[]
        if self.avg==True:
            for i in range(len(list_img)):
                M=np.mean(list_img[i],axis=-1)[:,:,:,None]
                L+=[M]
                
        else:
            L=list_img
        return L
      

    def mask_list_img(self,list_img,masker):
        list_img_masked=[]
        for i in range(len(list_img)):
            list_img_masked+=[nib.Nifti1Image(list_img[i],np.eye(4))]
        list_img_masked=masker.transform_imgs(list_img_masked)
        return np.array(list_img_masked).reshape(len(list_img),-1)
      
    def get_mask(self,list_img):
        mask=np.zeros_like(list_img[0][:,:,:,0])
        for i in range(len(list_img)):
            img=list_img[i]
            img=(img!=0).sum(axis=-1)
            mask+=(img!=0)
        mask=mask!=0
        mask=nib.Nifti1Image(mask.astype(float),np.eye(4))
        if self.normalisation=="time" and self.detrend==True:
            masker=MultiNiftiMasker(mask, standardize=True,detrend=True)
        elif self.normalisation=="time" and self.detrend==False:
            masker=MultiNiftiMasker(mask, standardize=True,detrend=False)
        elif self.detrend==True:
            masker=MultiNiftiMasker(mask, standardize=False,detrend=True)
        else:
            masker=MultiNiftiMasker(mask, standardize=False,detrend=False)
        masker.fit(list_img[0])
        self.masker=masker
    

    def standardize(self,list_img):
      """
      clean_img(nib.Nifti1Image(np.load(seq),np.eye(4)),ensure_finite=True,detrend=False
                                        #,t_r=2.5,high_pass=0.008,low_pass=0.08,**{"clean__butterworth__padlen":33}
                                        ).get_fdata().ravel() 
      """
      """
      no need to remove zeros since already masked
      """
      for i in range(len(list_img)):
          list_img[i]=list_img[i]-list_img[i].mean()
          list_img[i]=list_img[i]/list_img[i].std()
      return list_img

def get_string_to_remove(file_list, prefix="sub-"):
    for i in range(len(file_list[0])):
            if file_list[0][i:i+len(prefix)]==prefix:
                string_to_remove=file_list[0][:i+len(prefix)]
                break
    return string_to_remove

def average_prediction(weights,X,list_model,P_list):
    """
    average the prediction of all models
    weights:list of weight for each model
    X:data
    list_model:list of model to average prediction from
    P_list:list of preprocesser
    """
    weights=np.array(weights)
    results=[]
    for i,m in enumerate(list_model):#for each model
      X_=P_list[i].transform(X)#copy and transform data
      results+=[m.predict(X_)]#append prediction
    results=np.array(results).T.astype(int)#transpose so that each line is the different prediction for each sample
    L=[]
    for i in results:#for each sample
      pred=np.zeros(list_model[0].classes_.shape[0])#list of prediction set up to 0
      for classe in np.unique(i):#for each unique class predicted
        pred[classe]=np.sum(weights[i==classe])#compute the sum of the weights of the model having made that prediction
      L+=[pred]#append final averaged score for the sample
    return np.argmax(np.array(L),axis=-1)#return the index of the best score, ie: the class

def get_preprocessing_string(preprocessing_dic):
    if preprocessing_dic["detrend"]:
        detrend="_detrend"
    else:
        detrend=""
    if preprocessing_dic["average_time"]:
        avg_time="_average_time"
    else:
        avg_time=""
    if preprocessing_dic["thresh_PCA"]<=1:
        thresh_PCA="_"+str(preprocessing_dic["thresh_PCA"])+"%var_PCA"
    p=preprocessing_dic["normalisation"]
    return f"_{p}{avg_time}{detrend}{thresh_PCA}"