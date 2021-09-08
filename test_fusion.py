import torch
import cv2
import numpy as np
import torchvision
from dtdb_dataset import dtdb_dataset
from tqdm import tqdm

scaleFactor = [1,np.sqrt(2),2, 2*np.sqrt(2),4]

torch.set_grad_enabled(False)

class test_database(object):
    def __init__(self,model_rgb,model_dog,test_path,is_dog = True):
        self.model_rgb = model_rgb.to(torch.device('cuda:1'))
        self.model_dog = model_dog.to(torch.device('cuda:1'))
        self.test_path = test_path
        self.sequence_length = 25
        self.is_dog = is_dog
        self.test_set = dtdb_dataset(path = self.test_path,
                                     sequence_length = self.sequence_length,
                                     is_train = False,
                                     is_dog = self.is_dog,
                                     cuda_device = 'cuda:1')
        K = len(self.test_set.classes)
        
        # By seq
        self.matConf = np.zeros((K,K), dtype=np.int32)

        # By video
        self.nbVidsPerClass = np.zeros(K, dtype=np.int32)
        self.nbVidsPredictedPerClass = np.zeros(K, dtype=np.int32)
    
    def get_accuracy(self):
        print('Start getting accuracy!')
        correct = 0
        for i in range(len(self.test_set.lbs)):
            print('Getting accuracy for video:', i, '/', len(self.test_set.lbs), ': ', self.test_set.videos[i])
            prediction = self.predict_vid(i)
            correct += prediction
            print('correct = ', correct)
        return correct/len(self.test_set.lbs)
    
    def predict_vid(self,idx):
        vid_name = self.test_set.videos[idx]
        ground_truth = self.test_set.lbs[idx]
        correct = 0
        cap = cv2.VideoCapture(vid_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hw_ratio = frame_height/frame_width
        wh_ratio = frame_width/frame_height
        if frame_width > 256:
            frame_width = 256
            frame_height = int(hw_ratio * frame_width)
        
        if frame_height > 256:
            frame_height = 256
            frame_width = int(wh_ratio * frame_height)
        
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps > 30:
            downsample_rate = int(vid_fps / 25)
        else: 
            downsample_rate = 1
        
        num_blocks = int(frame_count/self.sequence_length-1)
        num_blocks_nonempty = 0
        for i in tqdm(range(num_blocks)):
            frame_block_DoG = np.empty((0,frame_height,frame_width,5))
            frame_block_RGB = np.empty((0,frame_height,frame_width,3))
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.sequence_length*i))
            fc = 0
            fc_1 = 0
            ret = True
            while fc_1<self.sequence_length and ret:
                ret,frame = cap.read()
                if fc%downsample_rate == 0 and np.array(frame).all() != None:
                    frame = cv2.resize(frame,(frame_width,frame_height),interpolation = cv2.INTER_LINEAR)
                    frame_RGB = frame.astype(np.float32)/255.0
                    frame_DoG = self.calcDog(frame_RGB)
                    frame_RGB = np.expand_dims(frame,axis=0)
                    frame_DoG = np.expand_dims(frame_DoG,axis=0)
                    frame_block_RGB = np.concatenate((frame_block_RGB,frame_RGB),axis=0)
                    frame_block_DoG = np.concatenate((frame_block_DoG,frame_DoG),axis=0)
                    fc_1 += 1
                fc += 1

#            frame_block = frame_block.astype(np.float32) / 255.0
            frame_block_RGB = torch.tensor(frame_block_RGB.transpose(3,0,1,2).astype(np.float32)/255.0, device = 'cuda:1')
            frame_block_RGB = frame_block_RGB.unsqueeze(0)
            frame_block_DoG = torch.tensor(frame_block_DoG.transpose(3,0,1,2).astype(np.float32), device = 'cuda:1')
            frame_block_DoG = frame_block_DoG.unsqueeze(0)
            if frame_block_RGB.shape[2]>0:
                num_blocks_nonempty += 1
                output = self.predict_clip(frame_block_RGB,frame_block_DoG)
                self.matConf[ground_truth, output]+=1
                if output==ground_truth:
                    correct += 1
                # else:
                    # print('output=', output, ' gt=', ground_truth)
        cap.release()
        if correct>=(num_blocks_nonempty//2):
            self.nbVidsPredictedPerClass[ground_truth]+=1
        print(correct, ' correct blocks out of', num_blocks_nonempty)
        return int(correct>=(num_blocks_nonempty//2))
        
    def predict_clip(self,sequence_rgb,sequence_dog):
        assert len(sequence_rgb.size())==5, "Input sequence has to be a tensor of exactly 5 dimensions NxCxDxHxW"
        assert len(sequence_dog.size())==5, "Input sequence has to be a tensor of exactly 5 dimensions NxCxDxHxW"
        predict_vect_rgb = self.model_rgb(sequence_rgb)
        predict_vect_dog = self.model_dog(sequence_dog)
        predict_vect = predict_vect_rgb + predict_vect_dog
        _,predict_class = torch.max(predict_vect.data,1)
        predict_class = predict_class.item() # data.tolist()
        return predict_class
    
    def calcDog(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dogImg = np.empty((image.shape[0],image.shape[1],0),dtype = np.float32)
        dog1 = -cv2.GaussianBlur(image,(0,0),scaleFactor[0])+image
        dog1[dog1<0]=0
        dog1 = np.expand_dims(dog1,axis = 2)
        dogImg = np.concatenate((dogImg,dog1),axis = 2)
        for k in range(len(scaleFactor)-1):
            blur1 = cv2.GaussianBlur(image,(0,0),scaleFactor[k])
            blur2 = cv2.GaussianBlur(image,(0,0),scaleFactor[k+1])
            tempDoG = blur1 - blur2 
            tempDoG[tempDoG<0]=0
            tempDoG = np.expand_dims(tempDoG,axis = 2)
            dogImg = np.concatenate((dogImg,tempDoG),axis = 2)
        return dogImg

device = torch.device('cuda:1')

model_rgb = torchvision.models.video.r2plus1d_18(num_classes = 18).to(device)
model_dog = torchvision.models.video.r2plus1d_18(num_classes = 18).to(device)
model_dog.stem[0] = torch.nn.Conv3d(5, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
path = '../../datasets/DTDB/BY_DYNAMIC_FINAL/'

state_dict_rgb = torch.load('./trained_models/dtdb_wts_R2p1D_RGB.pth', map_location='cuda:0')
state_dict_dog = torch.load('./trained_models/dtdb_wts_R2p1D_DoG.pth', map_location='cuda:0')

model_rgb.load_state_dict(state_dict_rgb['bestModelWts'])
model_dog.load_state_dict(state_dict_dog['bestModelWts'])
print('Model loaded')

test_data = test_database(model_rgb = model_rgb,
                          model_dog = model_dog,
                          test_path = path)

print('Test accuracy is:', test_data.get_accuracy())
np.savetxt('matconf.txt', test_data.matConf)
np.savetxt('nb_vidz_per_class.txt', test_data.nbVidsPerClass)
np.savetxt('nb_vidz_predicted_per_class.txt', test_data.nbVidsPredictedPerClass)
