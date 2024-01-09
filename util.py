import numpy as np
import os
import matplotlib.pyplot as plt
import Clip.clip
from torch import optim
import pickle
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import Clip.clip as clip
import torch
from torchvision.datasets import CocoCaptions
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pytorch_metric_learning import losses
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import string


''' Dataset related '''

def encode_dataset(model, device, Loader, batch_size):

    with torch.no_grad():
        image_to_text_map = []
        text_to_image_map = []

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0
        
        model = model.to(device)

        for images, text in tqdm(Loader):
            images = images.to(device)
            text = text.to(device)
         
            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape
            
            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                
                #index(map)
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            image_encodings.append(model.encode_image(images))
            text_encodings.append(model.encode_text(text))

        image_encodings = torch.cat(image_encodings).cpu()
        text_encodings = torch.cat(text_encodings).cpu()
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map

def get_map(Loader, device, batch_size):
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        text_index = 0
        image_index = 0

        for images, text in Loader:

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                
                #index(map)
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        return text_to_image_map, image_to_text_map

class Flickr30k(Dataset):
    """`Flickr30k Entities <https://bryanplummer.com/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(Flickr30k, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file) as fh:
            next(fh)
            for line in fh:
                img_id, _, captions = line.strip().split("|")
                self.annotations[img_id].append(captions)

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index] 

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        
        #wanna limit the size of target here but error happened when search relevant 
        target = self.remove_punctuation(target)
        target = self.limit_length(target)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.ids)
    
    def remove_punctuation(self,texts):

        punctuation = string.punctuation

        translator = str.maketrans('', '', punctuation)

        for i, text in enumerate(texts):
            texts[i] = text.translate(translator)

        return texts
    
    def limit_length(self, captions, max_length=50):
        # limit the length of caption
        return [' '.join(caption.split()[:max_length]) for caption in captions]

# continue encode dataset for proj layer
class encoded_dataset(Dataset):
    def __init__(self, image_encode, text_encode):
        self.image_encode = image_encode
        self.text_encode = text_encode
        
    def __getitem__(self, index):
        image = self.image_encode[index]
        text = self.text_encode[index*5: index*5+5]
        return image, text
    
    def __len__(self):
        return len(self.image_encode)

''' Loss related '''


class training_losses:
    def __init__(self, type, reduction='sum'):
        super(training_losses, self).__init__()
        self.type = type
        self.reduction = reduction
        self.temperature = 0.07
        self.ce_loss = nn.CrossEntropyLoss()
        self.ct_loss = losses.ContrastiveLoss()
    def forward(self, image_features, text_features, targets):
        tot_loss = None
        
        if self.type == 'corss_entropy':
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            image_loss = self.ce_loss(logits_per_image,targets)
            text_loss = self.ce_loss(logits_per_text, targets)
            tot_loss = (image_loss + text_loss)/2
            
        elif self.type == 'contrastive':
            logits_per_image = (image_features @ text_features.t()) / self.temperature
            logits_per_text = logits_per_image.t()
            
            image_loss = self.ct_loss(logits_per_image,targets)
            text_loss = self.ct_loss(logits_per_text, targets)
            tot_loss = (image_loss + text_loss)/2
            
        elif self.type == 'info_nce_loss':
            labels = torch.arange(image_features.size(0), device=image_features.device)
            logits = torch.matmul(image_features, text_features.T) / self.temperature
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = logits - logits_max  # for numerical stability
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            tot_loss = -log_prob.gather(1, labels.unsqueeze(1)).mean()
            
        elif self.type == 'nt_xent':
            logits_per_image = (image_features @ text_features.t()) / self.temperature
            logits_per_text = logits_per_image.t()

            image_loss = self.ce_loss(logits_per_image,targets)
            text_loss = self.ce_loss(logits_per_text, targets)
            tot_loss = (image_loss + text_loss)/2
            
        elif self.type == 'cos_embedd':
            cosine_similarity = F.cosine_similarity(image_features[:, None, :], text_features[None, :, :], dim=2)
            target = torch.eye(cosine_similarity.shape[0],dtype=torch.long, device=image_features.device)
            tot_loss = 0.5 * target * (1 - cosine_similarity) + 0.5 * (1 - target) * torch.clamp(cosine_similarity - 0.1, min=0.0)
            if self.reduction == 'sum':
                tot_loss = torch.sum(tot_loss)
            elif self.reduction == 'mean':
                tot_loss = torch.mean(tot_loss)    
                
        elif self.type == 'mix':
            logits_per_image = (image_features @ text_features.t()) / self.temperature
            logits_per_text = logits_per_image.t()

            image_loss = self.ce_loss(logits_per_image,targets)
            text_loss = self.ce_loss(logits_per_text, targets)
            ce_final_loss = (image_loss + text_loss)/2
            
            cosine_similarity = F.cosine_similarity(image_features[:, None, :], text_features[None, :, :], dim=2)
            target = torch.eye(cosine_similarity.shape[0],dtype=torch.long, device=image_features.device)
            cos_loss = 0.5 * target * (1 - cosine_similarity) + 0.5 * (1 - target) * torch.clamp(cosine_similarity - 0.1, min=0.0)
            if self.reduction == 'sum':
                cos_loss = torch.sum(cos_loss)
            elif self.reduction == 'mean':
                cos_loss = torch.mean(cos_loss)
                
            tot_loss = (ce_final_loss*100 + cos_loss)/2
            
        
        else:
            print('Wrong type!!!')
        
        return tot_loss

''' Model related '''


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# new projection layer
class proj_layer(nn.Module):
    def __init__(self, clips_model):
        super().__init__()
        self.text_proj = clips_model.text_projection
        self.image_proj = clips_model.visual.proj
        
    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = image @ self.image_proj
        text_features = text @ self.text_proj
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return image_features, text_features

class custom_adapter(nn.Module):
    def __init__(self, clips_model):
        super().__init__()
        self.text_proj = clips_model.text_projection
        self.image_proj = clips_model.visual.proj
        self.dtype = clips_model.dtype
        self.adapter = Adapter(512, 2).to(clips_model.dtype)
        self.ratio = 0.2
    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = image @ self.image_proj
        text_features = text @ self.text_proj
        
        # Adapter
        x = self.adapter(image_features)
        
        image_features = self.ratio * x + (1 - self.ratio) * image_features

        return image_features, text_features



def modified_structure(model_structure):
    # get transformer
    vision_transformer = model_structure.visual

    vision_transformer.proj = None  # modify the proj layer as None
    
    model_structure.text_projection = None
    

    return model_structure 



''' Initialization '''
def get_clip_model(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    version = args.model_version
    model, preprocess = clip.load(version, device=device, jit=False)

    return model, preprocess, device

def get_dataset(args):

    dataset_type = args.dataset
    bz = args.bz
    num_w = args.num_workers
    data_path = args.data_path

    #create the path to save the encoded data
    # on local
    # save_path = os.path.join(data_path, f'encoded_data/{dataset_type}/')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # on kaggle
    save_path = os.path.join(args.result_path, f'encoded_data/{dataset_type}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if args.data_type == 'raw':

        model, preprocess, device = get_clip_model(args)
        freeze = modified_structure(model)


        if dataset_type == 'moscoco':
            coco = CocoCaptions(root = os.path.join(data_path, 'train2017'),
                            annFile = os.path.join(data_path, 'annotations/captions_train2017.json'),
                            transform=preprocess,
                            target_transform=lambda texts: clip.tokenize(texts[0:5]))

            # 95% data for training，5% data for validation
            train_ratio = 0.95
            val_ratio = 0.05

            train_size = int(train_ratio * len(coco))
            val_size = len(coco) - train_size

            train, val = torch.utils.data.random_split(coco, [train_size, val_size])

            train_loader = DataLoader(coco_train,batch_size=bz,shuffle=False,num_workers=num_w,pin_memory=False)
            val_loader = DataLoader(coco_val,batch_size=bz,shuffle=Ture,num_workers=num_w,pin_memory=False)
        
        elif dataset_type == 'flickr':
            flick= Flickr30k(root=os.path.join(data_path, 'images'),
                            ann_file=os.path.join(data_path, 'captions.txt'), 
                            transform = preprocess, 
                            target_transform=lambda texts: clip.tokenize(texts[0:5]))
            dataset_len = len(flick)

            train_size, val_size, test_size= dataset_len-2000, 1000, 1000
            train_dataset, val_dataset, _ = torch.utils.data.random_split(flick, [train_size, val_size, test_size])

            train_loader = DataLoader(train_dataset,batch_size=bz,shuffle=False,num_workers=num_w,pin_memory=False)
            val_loader = DataLoader(val_dataset,batch_size=bz,shuffle=Ture,num_workers=num_w,pin_memory=False)
        
        if args.train_mode == 'total':
            # if we want to training all model structure, we return raw data
            return train_loader, val_loader
        
        image_train, text_train, _, _ = encode_dataset(freeze, device, train_loader, bz)
        image_val, text_val, _, _ = encode_dataset(freeze, device, val_loader, bz)

        # Save to a file using pickle
        with open(os.path.join(save_path, f'image_train.pkl'), 'wb') as file:
            pickle.dump(image_train, file)

        with open(os.path.join(save_path, f'text_train.pkl'), 'wb') as file:
            pickle.dump(text_train, file)

        with open(os.path.join(save_path, f'image_val.pkl'), 'wb') as file:
            pickle.dump(image_val, file)

        with open(os.path.join(save_path, f'text_val.pkl'), 'wb') as file:
            pickle.dump(text_val, file)

        train_encoded = encoded_dataset(image_train, text_train)
        train_encoded_loader = DataLoader(train_encoded, batch_size=bz, shuffle=True)

        val_encoded = encoded_dataset(image_val, text_val)
        val_encoded_loader = DataLoader(val_encoded, batch_size=bz, shuffle=True)


    elif data_type == 'encoded':

        #open the files are saved
        with open(os.path.join(data_path, f'image_train.pkl'), 'rb') as file:
            image_train = pickle.load(file)

        with open(os.path.join(data_path, f'text_train.pkl'), 'rb') as file:
            text_train = pickle.load(file)
            
        with open(os.path.join(data_path, f'image_val.pkl'), 'rb') as file:
            image_val = pickle.load(file)

        with open(os.path.join(data_path, f'text_val.pkl'), 'rb') as file:
            text_val = pickle.load(file)

        train_encoded = encoded_dataset(image_train, text_train)
        train_encoded_loader = DataLoader(train_encoded, batch_size=bz, shuffle=True)

        val_encoded = encoded_dataset(image_val, text_val)
        val_encoded_loader = DataLoader(val_encoded, batch_size=bz, shuffle=True)

    return train_encoded_loader, val_encoded_loader

def test_dataset(args):
    dataset_type = args.dataset
    bz = args.bz
    num_w = args.num_workers
    data_path = args.data_path

    #create the path to save the encoded data
    # on local
    # save_path = os.path.join(data_path, f'encoded_data/{dataset_type}/')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # on kaggle
    save_path = os.path.join(args.result_path, f'encoded_data/{dataset_type}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if args.data_type == 'raw':

        model, preprocess, device = get_clip_model(args)
        freeze = modified_structure(model)


        if dataset_type == 'moscoco':

            coco_test = CocoCaptions(root = os.path.join(data_path, 'val2017'),
                            annFile = os.path.join(data_path, 'annotations/captions_val2017.json'),
                            transform=preprocess,
                            target_transform=lambda texts: clip.tokenize(texts[0:5]))

            test_loader = DataLoader(coco_test,batch_size=bz,shuffle=False,num_workers=num_w,pin_memory=False)
        
        elif dataset_type == 'flickr':
            flick= Flickr30k(root=os.path.join(data_path, 'images'),
                            ann_file=os.path.join(data_path, 'captions.txt'), 
                            transform = preprocess, 
                            target_transform=lambda texts: clip.tokenize(texts[0:5]))
            dataset_len = len(flick)

            train_size, val_size, test_size= dataset_len-2000, 1000, 1000
            _, _, test_dataset = torch.utils.data.random_split(flick, [train_size, val_size, test_size])

            test_loader = DataLoader(test_dataset,batch_size=bz,shuffle=False,num_workers=num_w,pin_memory=False)
        
        if args.train_mode == 'total':
            # if we want to training all model structure, we return raw data
            return test_loader, None
        
        image_test, text_test, _, _ = encode_dataset(freeze, device, test_loader, bz)

        # Save to a file using pickle
        with open(os.path.join(save_path, f'image_test.pkl'), 'wb') as file:
            pickle.dump(image_test, file)

        with open(os.path.join(save_path, f'text_test.pkl'), 'wb') as file:
            pickle.dump(text_test, file)
            
        encoded_test = encoded_dataset(image_test, text_test)
        test_loader = DataLoader(encoded_test, batch_size=bz, shuffle=False)

    elif data_type == 'encoded':
    
        #open the files are saved
        with open(os.path.join(data_path, f'image_test.pkl'), 'rb') as file:
            image_test = pickle.load(file)

        with open(os.path.join(data_path, f'text_test.pkl'), 'rb') as file:
            text_test = pickle.load(file)
        
        encoded_test = encoded_dataset(image_test, text_test)

        test_loader = DataLoader(encoded_test, batch_size=bz, shuffle=False)

    return image_test, text_test, test_loader


def model_setup(args):
    clip_model, preprocess, device = get_clip_model(args)
    model_name = args.dataset + '_' + args.train_mode + '_' + args.model_version
    
    if args.resume:
        model_dir = os.path.join(args.model_path, f'{model_name}.pth')
        checkpoint = torch.load(model_dir)
        if args.train_mode == 'only_proj':
            model = proj_layer(clip_model)

        elif args.train_mode == 'with_adapter':
            model = custom_adapter(clip_model)

        elif args.train_mode == 'total':
            model = clip_model
        
        model.load_state_dict(checkpoint)

    else:
        if args.train_mode == 'only_proj':
            model = proj_layer(clip_model)

        elif args.train_mode == 'with_adapter':
            model = custom_adapter(clip_model)

        elif args.train_mode == 'total':

            model = clip_model

        return model, device

def train_setup(args):
    if args.task == 'zero_shot':
        model, device = model_setup(args)
        return model, device
    opt = args.optimizer
    learning_rate = args.lr
    scheduler = args.scheduler
    loss_type = args.loss_type
    model, device = model_setup(args)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if opt == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=learning_rate ,weight_decay=1e-3)
    elif opt == 'sgd':
        optimizer = optim.sgd(trainable_params, lr=learning_rate ,weight_decay=1e-3, momentum=0.9)
    elif opt == 'AdamW':
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, betas=(0.9,0.98), eps=1e-6,weight_decay=1e-3)
    
    if scheduler != None:
        if scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.6)
        elif scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3)
    else:
        lr_scheduler = None

    loss_func = training_losses(loss_type, reduction = 'sum')

    return model, device, optimizer, scheduler, loss_func

def plot_loss(train_loss_list, val_loss_list, args):
    max_epoch = args.max_epochs
    loss_type = args.loss_type
    plt.figure()
    plt.plot(range(max_epoch), train_loss_list, label='Training loss')
    plt.plot(range(max_epoch), val_loss_list, label='Validation loss')
    plt.xlabel('epoch')
    plt.title(loss_type)
    plt.legend()
    if args.sr:
        plt.savefig(path.os.join(args.result_path, f'{loss_type}_curve.png'))
    plt.show()
    plt.close()





    


    









    
