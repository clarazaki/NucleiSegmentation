#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.transforms.functional as tf

import os
import pandas as pd
from torchvision.io import read_image
import skimage.io as io

from PIL import Image
import cv2

import torch.optim as optim
from tqdm.notebook import tqdm

import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryJaccardIndex


# # 1. Creating Dataset
# 

# In[ ]:



class MyDataset(Dataset):
    def __init__(self, train_dir, transform_image=None, transform_mask=None): 
        
   
        self.train_dir = train_dir 
        self.folder = os.listdir(train_dir)
        self.folder.remove('.DS_Store') # remove hidden file from folder
        
        self.transform_image = transform_function_image
        self.transform_mask = transform_function_mask
                 
    def __getitem__(self, idx):
        
        
        images_folder = os.listdir(os.path.join(self.train_dir, self.folder[idx]))[1]
    
        masks_folder = os.listdir(os.path.join(self.train_dir, self.folder[idx]))[2]
        
        img_dir = os.path.join(self.train_dir, self.folder[idx], images_folder)    # loading data by index

        mask_dir = os.path.join(self.train_dir, self.folder[idx], masks_folder) 
        
        image = cv2.imread(os.path.join(img_dir, os.listdir(img_dir)[0]))
    
        image = Image.fromarray(image)
        

        if self.transform_image:
            image = self.transform_image(image)

    # creating the ground thruth masks by adding all masks for one image together
        mask = np.zeros((256, 256)) 
        mask = torch.from_numpy(mask)


        for i in range(0, len(os.listdir(mask_dir))):

            mask_sub = cv2.imread(os.path.join(mask_dir, os.listdir(mask_dir)[i]), 0)
            mask_sub=np.clip(mask_sub/255, 0.0, 1.0) # make sure msaks aare binary
         
            mask_sub = Image.fromarray(mask_sub)

            if self.transform_mask:
                mask_sub = self.transform_mask(mask_sub)

            mask = mask + mask_sub
        mask=np.clip(mask, 0.0, 1.0).float()
        
        return image.to(device), mask.to(device)
    
            
    def __len__(self):
        return len(self.folder) # length of training dataset = number of images in dataset
    


train_direction = '/Users/carla/Downloads/data-science-bowl-2018/stage1_train'

transform_function_image = transforms.Compose([
    transforms.Resize((256, 256)),  # resize the image
    transforms.ToTensor()  # convert the image to a tensor

])

transform_function_mask = transforms.Compose([
    transforms.Resize((256, 256)),  # resize the image
    transforms.ToTensor()  # convert the image to a tensor

])

dataset_train = MyDataset(train_direction, transform_image = transform_function_image, transform_mask = transform_function_mask)



# ### make one image visible to see dataset reads images the right way:

# In[ ]:


image, mask1 = dataset_train.__getitem__(3)

def display_img(img: torch.Tensor) -> None:
    plt.imshow(img.permute(1, 2, 0).cpu())


# # 2. Build U-Net architecture

# In[ ]:


class doubleConv(nn.Module):
    
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv  = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size = 3, stride = 1, padding = 0), # 1. convolution layer
            nn.ReLU(inplace=True), # activation layer
            nn.BatchNorm2d(channel_out), # Batch normaliztion
            nn.Conv2d(channel_out, channel_out, kernel_size = 3, stride = 1, padding = 0), # 2. convolution layer
            nn.ReLU(inplace=True),    
            nn.BatchNorm2d(channel_out)
        )
    
    
    def forward(self, x):
        
        return self.conv(x)
    

    
# define down for going downword on encoding and increassing feautere number (combi between maxPool and double conv)
    
class downscale(nn.Module):
    
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            doubleConv(channel_in, channel_out) # nested model (from above)
        )
    
    def forward(self, x):
        
        return self.max_pool(x)

    

class upscale(nn.Module):
    
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size = 2, stride = 2, padding = 0)
        self.conv2 = doubleConv(channel_in, channel_out)

    def forward(self, decode_tensor, encode_tensor):
        
    
        decode_tensor = self.deconv(decode_tensor)
        
      # pad tensors, so that skip connection arrays are the same size
    
        diffY = (encode_tensor.size()[2] - decode_tensor.size()[2]) 
        diffX = (encode_tensor.size()[3] - decode_tensor.size()[3])
        diffY = int(diffY)
        diffX = int(diffX)
        
        diffX_pad1 = int(diffX/2)
       
        diffX_pad2 = int(diffX-diffX/2)
        
        
        diffY_pad1 = int(diffY/2)
        
        diffY_pad2 = int(diffY-diffY/2)
       
        
        if diffY/2 - int(diffY/2) > 0  and diffX/2 - int(diffX/2) > 0:
            
            decode_tensor_pad = F.pad(decode_tensor, (diffX_pad1, int(diffX_pad2+1), diffY_pad1, int(diffY_pad2+1)))
            
        
        else:
            decode_tensor_pad = F.pad(decode_tensor, (diffX_pad1, diffX_pad2, diffY_pad1, diffY_pad2))
        
        x = torch.cat([encode_tensor, decode_tensor_pad], dim=1)
       
        
        return self.conv2(x)
class Up(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.up=nn.ConvTranspose2d(channels_in, channels_in//2, kernel_size=2, stride=2)
        self.conv=doubleConv(channels_in, channels_out) 

    def forward(self, x1, x2):
        x1=self.up(x1)
        diffY=x2.size()[2]-x1.size()[2]
        diffX=x2.size()[3]-x1.size()[3]
        
        x1=torch.nn.functional.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x=torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(OutConv, self).__init__()
        self.conv=nn.Conv2d(channels_in, channels_out, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    
    def __init__(self, channels_in, n_classes):
        super(UNet, self).__init__()

        self.firstConv = doubleConv(channels_in, 64) 
        self.down1 = downscale(64, 128)  
        self.down2 = downscale(128, 256)
        self.down3 = downscale(256, 512)
        self.down4 = downscale(512, 1024)
        
        
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.lastConv = OutConv(64, n_classes) 
        
        self.Sig = nn.Sigmoid()
        
    def forward(self, x): # x = image
        
        x1_beg = self.firstConv(x)
       
        x2_down = self.down1(x1_beg)
       
        x3_down = self.down2(x2_down)
        x4_down = self.down3(x3_down)
        
        x5_down = self.down4(x4_down)
       
        
        
        
        
        x = self.up1(x5_down, x4_down)
        x = self.up2(x, x3_down)
        x = self.up3(x, x2_down)
        x = self.up4(x, x1_beg)
        logits = self.lastConv(x)
        
        logits_sig = self.Sig(logits) # get predition values between 0 and 1 using sigmoid function
        mask_bin = (logits_sig >= 0.5).float()
        mask_bin.requires_grad=True	
        
        return logits.squeeze(1), logits_sig, mask_bin
# returns logits[0], prediction mask with values between 0 and 1[1], binary mask[2]


# # 3. Training U-Net

# ### Setting Hyperparameters and loading dataset

# In[ ]:


# constant hyperparameter

batchsize = 5
num_epochs = 20

learning_rate = 0.001

# load data for training
dataset_train = MyDataset(train_direction, transform_image = transform_function_image,  transform_mask = transform_function_mask)

dataloader_train = DataLoader(
    dataset = dataset_train, 
    batch_size = batchsize, 
    shuffle = 'True'
)


# ### Dice loss definition

# In[ ]:


# Dice Loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        
        inputs = inputs.squeeze(0) 
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        
        
        return 1 - dice
    

metric_Dice = DiceLoss()


# BCE Loss

metric_BCE = nn.BCELoss()


# ### Optimizers

# In[ ]:


# Adam
optimizer_adam = optim.Adam(model_Dice.parameters(), lr=learning_rate)
# SGD
optimizer_SGD = optim.SGD(model_Dice.parameters(), lr=learning_rate)


# ### Iteration loop for training 

# #### Here we show an examplary iteraation loop for training the model. We use the same loop, but change the metric for loss and the optimizers at different learning rates to compare 8 different models (dif. loss/optimizer/lr). 

# In[ ]:


# create model with certain Hyperparameters to train
model_Dice = UNet(channels_in = 3, n_classes = 1).to(device)


# metrics to show accuracy and iou for training process (similar to loss curves for training)
BA = BinaryAccuracy()
B_iou = BinaryJaccardIndex()

step_losses_Dice = []
epoch_losses_Dice = []

accur_all_epochs_Dice = []
IoU_value_all_epochs_Dice = []

model_Dice.train()
for epoch in tqdm(range(num_epochs)):
    epoch_loss_Dice = 0
    accur_epoch_Dice = 0
    IoU_value_epoch_Dice = 0
    
    model_Dice.train()
    for X, Y in tqdm(dataloader_train, total=len(dataloader_train)):
        optimizer.zero_grad()
        truth_mask = tf.center_crop(Y, [248, 248])
        try:
            assert truth_mask.min()>=0.0 and truth_mask.max()<=1.0
        except AssertionError:
            print("ASSERTON ERROR!")
            print("truth mask min: ", truth_mask.min())
            print("truth mask max: ", truth_mask.max())
            print("min: ", truth_mask.min()>=0.0)
            print("max: ", truth_mask.max()<=1.0)
 
        Y_pred = model_Dice(X)[1]
        Y_pred=Y_pred.squeeze().float()
        truth_mask=truth_mask.squeeze().float()
     
        loss = metric_Dice(Y_pred, truth_mask)
       
        loss.backward()
        optimizer_Dice.step()
    
    
        epoch_loss_Dice += loss.item()
        
        step_losses_Dice.append(loss.item())
        
        # Accuracy
        truth_mask = (truth_mask >= 0.5).float()
        accur_Dice = BA(Y_pred, truth_mask)
        accur_epoch_Dice += accur_Dice.item()
        
        # IoU
        IoU_value_Dice = B_iou(Y_pred, truth_mask)
        IoU_value_epoch_Dice += IoU_value_Dice.item()
        
        
    epoch_losses_Dice.append(epoch_loss_Dice/len(dataloader_train))
    accur_all_epochs_Dice.append(accur_epoch_Dice/len(dataloader_train))
    IoU_value_all_epochs_Dice.append(IoU_value_epoch_Dice/len(dataloader_train))
    
    print("EPOCH LOSS MEAN: ", epoch_loss_Dice/len(dataloader_train))
    print('EPOCH ACURRACY MEAN:', accur_epoch_Dice/len(dataloader_train))
    print('EPOCH IoU MEAN:', IoU_value_epoch_Dice/len(dataloader_train))


# ### arrays of loss, accuracy and IoU for plotting learning curves during training

# In[ ]:


# Learning rate e-3
# loss curve, accuracy curve and IoU curve for BCE+Adam trained model (on GPU)
model_BCE_Adam_trained_GPU_losscurve = np.array([0.3673354149498838, 0.22749005892175309, 0.1747550681391929, 0.14949620380363565, 0.14384550026598128, 0.12469681998358127, 0.1218663272546961, 0.11889105860857253,  0.14150564332908772, 0.11384211480617523, 0.09796188573570962, 0.1005962843947271, 0.09213209701107537, 0.08894925017623191, 0.08842880976326922, 0.0835183193867511, 0.08577423560571798, 0.08195604574173054, 0.08219757688013797, 0.08929298707145325])
model_BCE_Adam_trained_GPU_accuracycurve = np.array([0.9086571063132997, 0.9514411453236925, 0.9529763834273561, 0.9563718057693319, 0.9604199091170696, 0.9613317295591882, 0.9646694152913196, 0.963094321971244, 0.9590063367752318, 0.9630736340867713, 0.966514269088177, 0.9658816539226694, 0.9673271223585657, 0.9675968504966573, 0.9700239148545773, 0.9694756060204608, 0.968897300831815, 0.9701648196007343, 0.970107154643282, 0.9697884714349787])
model_BCE_Adam_trained_GPU_IoUcurve = np.array([0.5960964298945792, 0.7035932502848037, 0.7197339116258824, 0.7346816522643921, 0.7523518680258, 0.7590593365912742, 0.7795587714682234, 0.7698904932179349, 0.7481221176208334, 0.7746731521601372, 0.7861914843954938, 0.7892772713874249, 0.7914733052887815, 0.7962830073021828, 0.8032322941308326, 0.8029815789232863, 0.7993802437756924, 0.807289998582069, 0.8041213883998546, 0.8018799810967547])

# loss curve, accuracy curve and IoU curve for Dice+Adam trained model (on GPU)
model_Dice_Adam_trained_GPU_losscurve = np.array([0.41600712499719983, 0.26355667570804026, 0.20177472335226992, 0.19140041889028347, 0.1659963993316001, 0.201306015887159, 0.16341509907803636, 0.14216895623410003, 0.15698660941834144, 0.14013851703481472, 0.15261998709211957, 0.13293763741533807, 0.1340249206157441, 0.12359068812207973, 0.1232617237466447, 0.12398987881680752, 0.12132241878103703, 0.11693487712677489, 0.11786522763840696, 0.12854763167969724, ])
model_Dice_Adam_trained_GPU_accuracycurve  = np.array([0.841935137484936, 0.9225666377138584, 0.9427883599666839, 0.9488396213409749, 0.9560394997292376, 0.9457686860510643, 0.9560969941159512, 0.9619914730812641, 0.9590423107147217, 0.9625597767373348, 0.959436057095832, 0.9650063216686249, 0.9648199652103667, 0.9679031422797669, 0.9677369207777875, 0.967907985474201, 0.9684066569551508, 0.9691868158096962, 0.9682703563507568,  0.9666453092656238, ])
model_Dice_Adam_trained_GPU_IoUcurve = np.array([0.48907271534838576, 0.6236063324390574, 0.6860471840234513, 0.6994740760072748, 0.7315541442404402, 0.6861576853280372, 0.7343516612940646, 0.7667182896365511, 0.7461590208905808, 0.7697664109316278, 0.75023500843251, 0.7795068951363259, 0.77907949876278, 0.7946641099579791, 0.7945941921244276, 0.7931423586733798, 0.7977549687344977, 0.8048514787186968, 0.8028951179473958, 0.7856096612646225])

# loss curve, accuracy curve and IoU curve for BCE+SGD trained model (on GPU)
model_BCE_SGD_trained_GPU_losscurve = np.array([0.5569525684447999, 0.47506310014014547, 0.4416893422603607, 0.409667636168764, 0.38666343467032654, 0.3668492407874858, 0.3537540242392966, 0.3344821999681757, 0.3227359180120712, 0.31219881186459925, 0.2951015647738538, 0.28946333885826964, 0.2803620168186249, 0.2686390334621389, 0.26133912024979894, 0.25623842836060423, 0.2472462365601925, 0.24140381686230925, 0.23914707753252476, 0.2288944808409569])
model_BCE_SGD_trained_GPU_accuracycurve  = np.array([0.8174022249084838, 0.8799260943493945, 0.8979218779726231, 0.9130517035088641, 0.9215050885017882, 0.9293587588249369, 0.9337475731017741, 0.9389548111469188, 0.9393742959550087, 0.9439148395619494, 0.9470770606335174, 0.9504209718805678, 0.9503865857073601, 0.9532107907406827, 0.9533695115687999, 0.9549140537038763,  0.9555773912592137, 0.9565076206592803, 0.9560428515393683, 0.9576022707401438])
model_BCE_SGD_trained_GPU_IoUcurve = np.array([0.4148174900006741, 0.5251196437376611, 0.5652037788420281, 0.6022509997512432, 0.628422563618168, 0.6460591828886498, 0.6600260195579934, 0.6789705303121121, 0.6814636063385517, 0.6900280962916131, 0.7102397406037818, 0.7158653447602658, 0.7120510672635221, 0.7335142344236374, 0.7271499703539178, 0.730013334053628, 0.7357399371710229, 0.7359835381837602, 0.7378446757793427, 0.7468366058582955])

# loss curve, accuracy curve and IoU curve for Dice+SGD trained model (on GPU)
model_Dice_SGD_trained_GPU_losscurve = np.array([0.6440106452779567, 0.580442322061417, 0.5455885795836753, 0.5253668484535623, 0.5020084786922374, 0.4856777121411993, 0.4737949961043419, 0.46478872920604464, 0.447622057605297,  0.43688204504073935, 0.43159290387275373, 0.4250817355957437, 0.41078730783563977, 0.4057095342494072, 0.4012176876372479, 0.39271077767331547, 0.38105772530778925, 0.3718342698634939, 0.36776988239998515, 0.3645849988815632])
model_Dice_SGD_trained_GPU_accuracycurve  = np.array([0.7071513394091992, 0.744995688504361, 0.790200948081118, 0.8179902193394113, 0.8406065283937657, 0.8572280020155805, 0.8597534584238175, 0.8648067173805642, 0.8734956871955952, 0.8773717556862121, 0.8775766149480292, 0.8834759421805118, 0.8862608138551104, 0.8856157419529367, 0.8913687676825421, 0.8938845377019111, 0.8969589566930811, 0.899958351191054, 0.9050034094364086, 0.90304034948349])
model_Dice_SGD_trained_GPU_IoUcurve = np.array([0.3146404521737961, 0.36187751286048836, 0.4108695019274316, 0.4410444167145389, 0.47502697013477063, 0.49240489605259385, 0.5018476469719664, 0.5044267572304035, 0.5219567441876899, 0.5287416140608331, 0.531995934533312, 0.5354956670644435, 0.5457871787725611, 0.5463736025576896, 0.5539021216174389, 0.5619444672731643, 0.5691187678182379, 0.5740002431768052, 0.5849921070831887, 0.5816616333545522])



# Learning rate e-4 (Adam)
# loss curve, accuracy curve and IoU curve for BCE+Adam trained model (on GPU) with learning rate e-4
model_BCE_Adam_trained_GPU_losscurve_lr4 = np.array([0.45475316079373057, 0.3409540887842787, 0.3040266579135935, 0.2625140114984614, 0.25434179493087405, 0.23416104865200976, 0.2239909808527916, 0.20353020489849943, 0.19114818527026378, 0.1860201929477935, 0.18804410131687813, 0.17982593582982712, 0.16285334194594242, 0.16094172793499967, 0.14079399502023737, 0.155508502366695, 0.13958207565419217, 0.13972511824141157, 0.13305147368698678, 0.13269470532999394])
model_BCE_Adam_trained_GPU_accuracycurve_lr4 = ([0.8873338807136455, 0.9494727932392283, 0.9534898279829228, 0.960218570333846, 0.9619562892203636, 0.96208079064146, 0.9641203036967744, 0.9661621749401093, 0.9658114922807571, 0.9650926716784214, 0.966266505895777, 0.966868407548742, 0.9690385809604157, 0.9680818883662529, 0.9704692122784067, 0.9706836583766532, 0.9709292028812652, 0.971504917804231, 0.9701990003281451, 0.9703877764813443])
model_BCE_Adam_trained_GPU_IoUcurve_lr4 = ([0.5921015504826891, 0.7172768735029595, 0.725819098029999, 0.7567478554679993, 0.7640547864931695, 0.7607500039516611, 0.7747837314580349, 0.7859231398460713, 0.7843025285512844, 0.7807831275970378, 0.7866461552838062, 0.7872878918622402, 0.8064435244874751, 0.7936145013317148, 0.8090761579731678, 0.8119146437086957, 0.8147458294604687, 0.8161579738271997, 0.8061329059778376, 0.808049737773043])

# loss curve, accuracy curve and IoU curve for Dice+Adam trained model (on GPU)
model_Dice_Adam_trained_GPU_losscurve_lr4 = np.array([0.5103955934656427, 0.4047325335918589, 0.35213201540581723,  0.3011225677551107, 0.26683398320319807, 0.23206794769205946, 0.21445256027769535, 0.20707880753151914, 0.18359092765666069, 0.17499868666872065,0.17594403535761732, 0.16116561344329347, 0.141184612157497, 0.13926946926624217, 0.1457224982850095, 0.14946045900912994, 0.15629551638948155, 0.13263428972122518, 0.1411889284215075, 0.12849152975894035])
model_Dice_Adam_trained_GPU_accuracycurve_lr4  = np.array([0.845524732736831, 0.905490111163322, 0.9279592956634278, 0.9407568700770115, 0.9485801616881756, 0.9563534012500275, 0.9595369658571609, 0.9597356053108864, 0.9647629901449731, 0.9641236624819167, 0.9618929814785084, 0.9663738874678917, 0.9688608741506617, 0.9691977152164947, 0.9670037367242448, 0.9654544858222313, 0.9657747225558504, 0.9706714127926116, 0.9673708461700602, 0.970001847820079])
model_Dice_Adam_trained_GPU_IoUcurve_lr4 = np.array([0.4819253366044227,  0.5864400552942398, 0.6399457993659567,  0.6866689400470003, 0.7136235700008717, 0.743771013744334, 0.7578508096172455, 0.7547265249998012, 0.7772906831604369, 0.7771444455423253, 0.7661297463990272, 0.7866840638378834,  0.8065013245065161, 0.8062602674707453, 0.7920151425802961,  0.7854681503265462, 0.7784742922224896, 0.8099792133620445, 0.7904022913029853, 0.8054065349254202])



# Learning rate e-2 (SGD)
# loss curve, accuracy curve and IoU curve for BCE+SGD trained model (on GPU) with learning rate e-2

model_BCE_SGD_trained_GPU_losscurve_lr2 = np.array([0.40798089431321366, 0.27902158016854145, 0.23419471069219264, 0.2091284400605141, 0.19457696195929608, 0.1706697238569564, 0.16441818841911376, 0.153464675108169, 0.1464378075238238, 0.1419401149007868, 0.1331303807252899, 0.13087980540350397, 0.12849575954865902, 0.12930538366608163, 0.13712702485475134, 0.12348206341266632, 0.13561676367324718, 0.11352558893726226, 0.12144238561233307, 0.13262099202008956])
model_BCE_SGD_trained_GPU_accuracycurve_lr2  = np.array([0.8931496897910504, 0.947573795597604, 0.9530649800249871, 0.9543478920104655, 0.95360703290777, 0.9613322171759098, 0.9596604356106292, 0.9631989002227783, 0.9647783503887501, 0.9649408732322936, 0.9653252432955072, 0.9675981288260602, 0.9657362585372113, 0.9679002038975979, 0.9657021101484907, 0.9673388473531033, 0.966321199498278, 0.9667982864887157, 0.9675283571507068, 0.9661006667512528])
model_BCE_SGD_trained_GPU_IoUcurve_lr2 = np.array([0.5731194246005504,0.6995541164849667, 0.7191159423995526, 0.7278449560733552, 0.7234076469185504, 0.7598778342312955, 0.750034195945618, 0.7682992911085169, 0.7755045947876382, 0.776089534480521, 0.7798197072871188, 0.7947904543673738, 0.7847909049150792, 0.7940068000808675, 0.784615067091394, 0.7907655448355573, 0.7838108447637964, 0.7869175919826995, 0.7899318876418662, 0.7891629664821828])


model_Dice_SGD_trained_GPU_losscurve_lr2 = np.array([0.5170047327559045, 0.40755817864803556, 0.33959392854507936, 0.30011153728403944, 0.2810369273449512, 0.256479098441753, 0.22417777713308942, 0.21526537803893395, 0.190159761525215, 0.16492702859513303, 0.16937752797248515, 0.16064457056370188, 0.16278578689757814, 0.15595357595606052, 0.13990731695865063, 0.13341171944395025, 0.13157709862323516, 0.13624150068201918, 0.13031799679106854, 0.14317475861691414])
model_Dice_SGD_trained_GPU_accuracycurve_lr2 = np.array([0.8334283752644316, 0.8844295537218134, 0.9068171660950843, 0.9194081831485668, 0.9297546478027993, 0.9385704005018194, 0.9443988419593649, 0.9503995934699444, 0.9565794721562811, 0.9619626561377911, 0.9613176539857337, 0.9626498945215916, 0.963034026166226, 0.9637018930404744, 0.9662220535126138, 0.9683141847874256, 0.9679577654980599, 0.9677637631588794, 0.9676761379901399, 0.9668230547549876])
model_Dice_SGD_trained_GPU_IoUcurve_lr2 = np.array ([0.45788475775972326, 0.5377158524983741, 0.5896377452510468, 0.6238118273780701, 0.6457540862103726, 0.6735254361274394, 0.703487125958534, 0.7205074305864091, 0.7402225279427589, 0.7668446873096709, 0.7617618449507876, 0.7697739404566745, 0.7651954917831624, 0.770952600113889, 0.7888109002341616, 0.7976644070858651, 0.7974511175713641, 0.7970128091091805, 0.7985019341428229, 0.7848720309582162])



# In[ ]:


x_ = np.linspace(1, 20, 20)
plt.figure()
plt.figure(figsize=(10,8))
plt.plot(x_, model_BCE_Adam_trained_GPU_losscurve, 'cornflowerblue')
plt.plot(x_, model_BCE_Adam_trained_GPU_losscurve_lr4, 'cornflowerblue', linestyle='dashed')
plt.plot(x_, model_BCE_SGD_trained_GPU_losscurve, 'sandybrown')
plt.plot(x_, model_BCE_SGD_trained_GPU_losscurve_lr2, 'sandybrown', linestyle='dashed')

plt.plot(x_, model_Dice_Adam_trained_GPU_losscurve, 'navy')
plt.plot(x_, model_Dice_Adam_trained_GPU_losscurve_lr4, 'navy', linestyle='dashed')
plt.plot(x_, model_Dice_SGD_trained_GPU_losscurve, 'firebrick')
plt.plot(x_, model_Dice_SGD_trained_GPU_losscurve_lr2, 'firebrick', linestyle='dashed')


plt.title('Loss curve for training U-Net model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['BCE Loss + Adam Optimizer, lr e-3', 'BCE Loss + Adam Optimizer, lr e-4' ,'BCE Loss + SGD Optimizer, lr e-3', 'BCE Loss + SGD Optimizer, lr e-2','Dice Loss + Adam Optimizer, lr e-3', 'Dice Loss + Adam Optimizer, lr e-4','Dice Loss + SGD Optimizer, lr e-3', 'Dice Loss + SGD Optimizer, lr e-2'])
plt.show()


# In[ ]:


plt.figure()
plt.figure(figsize=(10,8))
plt.plot(x_, model_BCE_Adam_trained_GPU_accuracycurve, 'cornflowerblue')
plt.plot(x_, model_BCE_Adam_trained_GPU_accuracycurve_lr4, 'cornflowerblue', linestyle='dashed')
plt.plot(x_, model_BCE_SGD_trained_GPU_accuracycurve, 'sandybrown')
plt.plot(x_, model_BCE_SGD_trained_GPU_accuracycurve_lr2, 'sandybrown', linestyle='dashed')

plt.plot(x_, model_Dice_Adam_trained_GPU_accuracycurve, 'navy')
plt.plot(x_, model_Dice_Adam_trained_GPU_accuracycurve_lr4, 'navy', linestyle='dashed')
plt.plot(x_, model_Dice_SGD_trained_GPU_accuracycurve, 'firebrick')
plt.plot(x_, model_Dice_SGD_trained_GPU_accuracycurve_lr2, 'firebrick', linestyle='dashed')

plt.title('Accuracy curve for training U-Net model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['BCE Loss + Adam Optimizer, lr e-3', 'BCE Loss + Adam Optimizer, lr e-4' ,'BCE Loss + SGD Optimizer, lr e-3', 'BCE Loss + SGD Optimizer, lr e-2','Dice Loss + Adam Optimizer, lr e-3', 'Dice Loss + Adam Optimizer, lr e-4','Dice Loss + SGD Optimizer, lr e-3', 'Dice Loss + SGD Optimizer, lr e-2'])
plt.savefig('Accuracycurves_model_comparison.png')
plt.show()



# In[ ]:


plt.figure()
plt.figure(figsize=(10,8))

plt.plot(x_, model_BCE_Adam_trained_GPU_IoUcurve, 'cornflowerblue')
plt.plot(x_, model_BCE_Adam_trained_GPU_IoUcurve_lr4, 'cornflowerblue', linestyle='dashed')
plt.plot(x_, model_BCE_SGD_trained_GPU_IoUcurve, 'sandybrown')
plt.plot(x_, model_BCE_SGD_trained_GPU_IoUcurve_lr2, 'sandybrown', linestyle='dashed')

plt.plot(x_, model_Dice_Adam_trained_GPU_IoUcurve, 'navy')
plt.plot(x_, model_Dice_Adam_trained_GPU_IoUcurve_lr4, 'navy', linestyle='dashed')
plt.plot(x_, model_Dice_SGD_trained_GPU_IoUcurve, 'firebrick')
plt.plot(x_, model_Dice_SGD_trained_GPU_IoUcurve_lr2, 'firebrick', linestyle='dashed')

plt.title('IoU metric curve for training U-Net model')
plt.xlabel('Epoch')
plt.ylabel('IoU value')
plt.legend(['BCE Loss + Adam Optimizer, lr e-3', 'BCE Loss + Adam Optimizer, lr e-4' ,'BCE Loss + SGD Optimizer, lr e-3', 'BCE Loss + SGD Optimizer, lr e-2','Dice Loss + Adam Optimizer, lr e-3', 'Dice Loss + Adam Optimizer, lr e-4','Dice Loss + SGD Optimizer, lr e-3', 'Dice Loss + SGD Optimizer, lr e-2'])
plt.savefig('IoUcurves_model_comparison.png')
plt.show()


# # 4. Validation

# ### load trained models

# In[ ]:


model_BCE_Adam_trained_GPU = torch.load('/Users/carla/Downloads/BCE.pt', map_location=torch.device('cpu'))
model_Dice_Adam_trained_GPU = torch.load('/Users/carla/Downloads/Dice.pt', map_location=torch.device('cpu'))
model_BCE_SGD_trained_GPU = torch.load('/Users/carla/Downloads/BCE_sgd.pt', map_location=torch.device('cpu'))
model_Dice_SGD_trained_GPU = torch.load('/Users/carla/Downloads/Dice_sgd.pt', map_location=torch.device('cpu'))

model_BCE_Adam_trained_GPU_lr4 = torch.load('/Users/carla/Downloads/BCE_adam_lr4.pt', map_location=torch.device('cpu'))
model_Dice_Adam_trained_GPU_lr4 = torch.load('/Users/carla/Downloads/Dice_adam_lr4.pt', map_location=torch.device('cpu'))
model_BCE_SGD_trained_GPU_lr2 = torch.load('/Users/carla/Downloads/BCE_SGD_lr2.pt', map_location=torch.device('cpu'))
model_Dice_SGD_trained_GPU_lr2 = torch.load('/Users/carla/Downloads/Dice_SGD_lr2.pt', map_location=torch.device('cpu'))



# ### import evaluation metric functions

# In[ ]:


# evaluation of the trained models

# evaluation metrics: precision, recall, accuracy, IoU value

#Recall:
from torchmetrics.classification import BinaryRecall
recall = BinaryRecall()


# Precision:
from torchmetrics.classification import BinaryPrecision
precision = BinaryPrecision()
 

# IoU value:
from torchmetrics.classification import BinaryJaccardIndex
IoU = BinaryJaccardIndex()


# Accurracy:
from torchmetrics.classification import BinaryAccuracy
accuracy = BinaryAccuracy()

# F1 score:
from torchmetrics.classification import BinaryF1Score
F1 = BinaryF1Score()


# ### load dataset for validation and predict mask for every image + calculate average precision, recall, IoU and accuracy

# #### here we show an exemplary validation loop to calculate the evaluation metrics for one trained model. We aapplied the same loop for all 8 hyperparameter combinations

# In[ ]:


# evaluation Dice+Adam trained model
batchsize = 1
validation1_direction = '/Users/carla/Downloads/data-science-bowl-2018/stage1_validate'
dataset_validate1 = MyDataset(validation_direction, transform_image = transform_function_image,  transform_mask = transform_function_mask)

dataloader_validate1 = DataLoader(
    dataset = dataset_validate1, 
    batch_size = batchsize, 
    shuffle = 'True'
)

recall_sum = 0
precision_sum = 0
IoU_sum = 0
accuracy_sum = 0

model_Dice_Adam_trained_GPU.eval()
for X, Y in tqdm(dataloader_validate1, total=len(dataloader_validate1)):
    
    with torch.no_grad():
        Y_pred = model_Dice_Adam_trained_GPU(X)[1]
    Y_pred = Y_pred.squeeze().float()

    truth_mask = tf.center_crop(Y, [248, 248])
    truth_mask = truth_mask.squeeze().float()
    truth_mask = (truth_mask >= 0.5).float()
 
    
    #Recall:
    recall_val = recall(Y_pred, truth_mask)
    recall_sum += recall_val.item()
    # Precision:
    precision_val = precision(Y_pred, truth_mask)
    precision_sum += precision_val.item()
    # IoU value:
    IoU_val = IoU(Y_pred, truth_mask)
    IoU_sum += IoU_val.item()
    # Accurracy:
    accuracy_val = accuracy(Y_pred, truth_mask)
    accuracy_sum += accuracy_val.item()
    
    
recall_mean = recall_sum/80
precision_mean = precision_sum/80
IoU_mean = IoU_sum/80
accuracy_mean = accuracy_sum/80

print('RECALL MEAN: ', recall_mean)
print('PRRECISION MEAN: ', precision_mean)
print('IoU MEAN: ', IoU_mean)
print('ACCURACY MEAN: ', accuracy_mean)





# ### Precision recall curve

# #### Again we show one exemplary calculation of the precision recall curve for the BCE+Adam at a lr of e-4 in this case. We used the same commands for calculating the precision recall curve for  BCE+Adam at a lr of e-3 and  Dice+Adam at a lr of e-4.

# In[ ]:



# flatten and concatenate all predicted and ground truth masks along dim 1

Y_pred_flat = np.array([])
truth_mask_flat = np.array([])

model_BCE_Adam_trained_GPU_lr4.eval()
for X, Y in tqdm(dataloader_validate1, total=len(dataloader_validate1)):
    
    with torch.no_grad():
        Y_pred = model_BCE_Adam_trained_GPU_lr4(X)[1]
    Y_pred = Y_pred.squeeze().float()

    truth_mask = tf.center_crop(Y, [248, 248])
    truth_mask = truth_mask.squeeze().float()
    truth_mask = (truth_mask >= 0.5).float()
 
    Y_pred_flat = np.append(Y_pred_flat, Y_pred)
    truth_mask_flat = np.append(truth_mask_flat, truth_mask)
    
print(Y_pred_flat.shape)
print(truth_mask_flat.shape)

# in the end, the shape of both aarrays should be #images * #x_pixels * #y_pixels


# In[ ]:


# calculate and plot precision-recll curve

from sklearn.metrics import precision_recall_curve

precision_metric, recall_metric, thresholds_metric = precision_recall_curve(truth_mask_flat, Y_pred_flat)



plt.figure()
plt.figure(figsize=(10,8))
plt.plot(recall_metric, precision_metric, color = 'sandybrown')
x = np.linspace(1, 0, 10)
y = np.linspace(1, 0, 10)
plt.plot(x, y[::-1], color = 'cornflowerblue', linestyle = 'dashed')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(['BCE+Adam, learning rate e-4'])
plt.show()


# # 5. Test 

# ### precision, recall, IoU and accuracy for test set

# In[ ]:


# load test dataset

batchsize = 1
test_direction = '/Users/carla/Downloads/data-science-bowl-2018/stage1_test'
dataset_test = MyDataset(test_direction, transform_image = transform_function_image,  transform_mask = transform_function_mask)

dataloader_test = DataLoader(
    dataset = dataset_test, 
    batch_size = batchsize, 
    shuffle = 'True'
)



recall_sum = 0
precision_sum = 0
IoU_sum = 0
accuracy_sum = 0

precison_array_test = []
recall_array_test = []
IoU_array_test = []

model_BCE_Adam_trained_GPU_lr4.eval()
for X, Y in tqdm(dataloader_test, total=len(dataloader_test)):
    
    with torch.no_grad():
        Y_pred = model_BCE_Adam_trained_GPU_lr4(X)[1]
    Y_pred = Y_pred.squeeze().float()

    truth_mask = tf.center_crop(Y, [248, 248])
    truth_mask = truth_mask.squeeze().float()
    truth_mask = (truth_mask >= 0.5).float()
 
    
    #Recall:
    recall_val = recall(Y_pred, truth_mask)
    recall_sum += recall_val.item()
    # Precision:
    precision_val = precision(Y_pred, truth_mask)
    precision_sum += precision_val.item()
    # IoU value:
    IoU_val = IoU(Y_pred, truth_mask)
    IoU_sum += IoU_val.item()
    # Accurracy:
    accuracy_val = accuracy(Y_pred, truth_mask)
    accuracy_sum += accuracy_val.item()
    
    precison_array_test.append(precision_val.item())
    recall_array_test.append(recall_val.item())
    IoU_array_test.append(IoU_val.item())
    
    
recall_mean = recall_sum/120
precision_mean = precision_sum/120
IoU_mean = IoU_sum/120
accuracy_mean = accuracy_sum/120

print('RECALL MEAN: ', recall_mean)
print('PRRECISION MEAN: ', precision_mean)
print('IoU MEAN: ', IoU_mean)
print('ACCURACY MEAN: ', accuracy_mean)





# ### precision recall curve + average precision

# In[ ]:


Y_pred_flat_test = np.array([])
truth_mask_flat_test = np.array([])

model_BCE_Adam_trained_GPU_lr4.eval()
for X, Y in tqdm(dataloader_test, total=len(dataloader_test)):
    
    with torch.no_grad():
        Y_pred = model_BCE_Adam_trained_GPU_lr4(X)[1]
    Y_pred = Y_pred.squeeze().float()

    truth_mask = tf.center_crop(Y, [248, 248])
    truth_mask = truth_mask.squeeze().float()
    truth_mask = (truth_mask >= 0.5).float()
 
    Y_pred_flat_test = np.append(Y_pred_flat_test, Y_pred)
    truth_mask_flat_test = np.append(truth_mask_flat_test, truth_mask)
    
print(Y_pred_flat_test.shape)
print(truth_mask_flat_test.shape)

print(average_precision_score(truth_mask_flat_test, Y_pred_flat_test))


# # 6. Visualization of FP, FN, TP, TN

# In[ ]:


validation_direction = '/Users/carla/Downloads/data-science-bowl-2018/stage1_validate'
dataset_validate_ROC = MyDataset(validation_direction, transform_image = transform_function_image,  transform_mask = transform_function_mask)

test_image, test_mask1 = dataset_train.__getitem__(5) # visualize TP, FP, FN, TN for iimaage 5 of validation dataset


test_mask1 = test_mask1.squeeze().float()
test_mask1 = (test_mask1 >= 0.5).float()
test_mask1 = tf.center_crop(test_mask1, [248, 248])


test_image = test_image.unsqueeze(0)

model_BCE_Adam_trained_GPU_lr4.eval()
with torch.no_grad():
    pred_mask = model_BCE_Adam_trained_GPU_lr4(test_image)[1] # get predicted mask
    pred_mask_prob = model_BCE_Adam_trained_GPU_lr4(test_image)[1]
    

pred_mask_prob = pred_mask_prob.squeeze(0)
pred_mask_prob = pred_mask_prob.squeeze(0)
pred_mask_prob = pred_mask_prob.cpu().detach().numpy() 
    
pred_mask = pred_mask.squeeze(0)
pred_mask = pred_mask.squeeze(0)
pred_mask = (pred_mask >= 0.5).float() # binarize predicted mask


pred_mask = pred_mask.cpu().detach().numpy()
test_mask1 = test_mask1.cpu().detach().numpy()

nucleus = pred_mask + test_mask1 #. aadding two binary masks --> resulting mask will consist of ones (FP, FN), zeros (TN) and twos (TP)
nucleus[nucleus!=2] = 0 # after summing two binary images the TN will be the values in the sum which are 0+0 = 0
nucleus[nucleus==2] = 1.7 # after summing two binary images the TP will be the values in the sum which are 1+1 = 2

    
TP = pred_mask == test_mask1 
TP = TP.astype(int)
print('a',  np.max(TP))
TP[TP == 1] = 0 


FP = pred_mask > test_mask1 
FP = FP.astype(float)
FP[FP == 1] = 2.2 # set all values where pred mask predicts nucleus (=1) and grount truth mask doesnt (=0), which is equal two all pixels where pred mask has higher value than ground truth mask, to a certain value (=FP)


FN = pred_mask < test_mask1 
FN = FN.astype(float)
FN[FN == 1] = 2.7 # set all values where pred mask doesn't predicts nucleus (=0) and grount truth mask does (=1), which is equal two all pixels where pred mask has lower value than ground truth mask, to a certain value (=FN)




color_mask = TP+FP+FN+nucleus # (add all masks to get certain value (0, 1.7 (TP), 2.2 (FP), 2.7 (FN)) at each region of resulting image)

color_mask[0, 0] = 3 # only to get dynamic raange of colors right


io.imshow(color_mask, cmap='nipy_spectral') # ni this color legend, TP will appear green, FP will appear yellow, FN will aappear red and TN will appear black


# # 7. Calculate histogramm of intesity distribution in ground truth masks to evaluate class imbalance

# In[ ]:


trainhist_direction = '/Users/carla/Downloads/data-science-bowl-2018/stage1_train'
dataset_trainhist = MyDataset(trainhist_direction, transform_image = transform_function_image,  transform_mask = transform_function_mask)

dataloader_trainhist = DataLoader(
    dataset = dataset_trainhist, 
    batch_size = batchsize, 
    shuffle = 'True'
)


Y_all = np.array([])

model_BCE_Adam_trained_GPU_lr4.eval()

for X, Y in tqdm(dataloader_trainhist, total=len(dataloader_trainhist)):
    
    with torch.no_grad():     
        Y.flatten
     
        Y_all = np.append(Y_all, Y)
      


# In[ ]:


plt.figure()
plt.figure(figsize=(10,8))
plt.hist(Y_all, bins=[-.5,.5,1.5], color = "skyblue", ec='steelblue')
plt.xticks((0,1))
plt.xlabel('Intensity')
plt.ylabel('Count')
plt.title('Histogramm of intensities of ground truth masks')
plt.show()  

