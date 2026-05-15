import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImageFolderDataset(Dataset):
    def __init__(self,root,transform=None):
        super(ImageFolderDataset,self).__init__()

        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg' , '.png' , '.jpeg'))]

    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root,self.files[index])
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        return img
    


def get_transforms(size,crop, final_size):
    transform_list = []
    if size>0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))
    
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)





def AdaIN(c_feature,s_feature):
    #[batch_size , channel , H, W]
    size = c_feature.size()
    style_mean ,style_std = cal_mean_std(s_feature)
    content_mean, content_std = cal_mean_std(c_feature)
    normalized_content_feat = (c_feature - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


def cal_mean_std(features,eps=1e-5):
    #[batch_size , channel , H, W]
    size = features.size()
    assert(len(size)==4)
    batch_size , channels = size[:2]
    feat_mean = features.view(batch_size, channels, -1).mean(dim=2).view(batch_size,channels,1,1)
    feat_var = features.view(batch_size, channels, -1).var(dim=2 , unbiased = False) + eps
    feat_std = feat_var.sqrt().view(batch_size,channels,1,1)
    return feat_mean , feat_std
