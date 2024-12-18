import torch, torch.nn as nn, torch.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
  
device = "cuda" if torch.cuda.is_available() else "cpu"
  
image_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
  
mask_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
])
  
class PetsDataset(Dataset):
    def __init__(self, dataset_path):
        self.image_paths = sorted(list(Path(dataset_path / "images").iterdir()))  # list of paths to individual images
        self.map_paths = sorted(
            list(Path(dataset_path / "trimaps").iterdir()))  # list of paths to individual annotation files
  
        ## Get dog vs cat classes from the image paths 
        self.species_names = ["dog", "cat"]  # list of species names
        self.species_classes = [image_path.name[0].isupper() * 1 for image_path in
                                self.image_paths]  # corresponding class idx for each image
  
        ### Add the breed names, breed classes and breed name to index mapping
        self.dog_breed_names = ['american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer',
                                'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
                                'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
                                'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
                                'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier',
                                'yorkshire_terrier']
        self.cat_breed_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
                                'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']
        self.breed_names = self.dog_breed_names + self.cat_breed_names
        self.breed_name2idx = {breed: idx for idx, breed in enumerate(self.breed_names)}
        self.breed_classes = [self.breed_name2idx["_".join(image_path.stem.split("_")[:-1])] for image_path in
                              self.image_paths]
  
        assert len(self.image_paths) == len(
            self.species_classes), f"Number of images and species_classes do not match: {len(self.image_paths)} != {len(self.species_classes)}"
        assert len(self.image_paths) == len(
            self.breed_classes), f"Number of images and breeds do not match: {len(self.image_paths)} != {len(self.breed_classes)}"
  
    def __len__(self):
        return len(self.image_paths)
  
    def __getitem__(self, idx):
        # 1. load the image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image_transform(image)
  
        # 2.load the mask
        mask = Image.open(self.map_paths[idx])
        mask = mask_transform(mask)
        mask = torch.tensor(np.array(mask)).long() - 1
  
        # 3. class tensors for classification
        species_tensor = torch.tensor(self.species_classes[idx])
        breed_tensor = torch.tensor(self.breed_classes[idx])
  
        return image, species_tensor, breed_tensor, mask
  
  
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   nn.Conv2d(out_ch, out_ch, kernel, padding=padding),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU())
  
    def forward(self, x):
        return self.layer(x)
  
  
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.ReLU())
  
    def forward(self, x):
        return self.layer(x)
 
class Encoder(nn.Module):
    def __init__(self, chs=(3,16,32,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)
     
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs, x
 
 
class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32, 16)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
         
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x
 
         
# Unet architecture for segmentation
class Net(nn.Module):
    def __init__(self,
                 enc_chs=(3,16,32,64,128,256), 
                 dec_chs=(256, 128, 64, 32, 16), 
                 num_class=3, 
                 image_sz=128,
                 mlp_base=256,
                 mlp_dims=[4, 3],
                 num_species=2,
                 num_breeds=37
                  ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)

        # Build multi layer perceptron for classification
        def build_mlp(out_dim):
            mlp_in = enc_chs[-1] * (image_sz // 2 ** (len(enc_chs)-1)) ** 2  # Flatten dim
            mlp_layers = [MLPBlock(mlp_in, mlp_base * mlp_dims[0])]
            for i in range(len(mlp_dims) - 1):
                mlp_layers.append(MLPBlock(mlp_base * mlp_dims[i], mlp_base * mlp_dims[i + 1]))
            mlp_layers.append(nn.Linear(mlp_base * mlp_dims[-1], out_dim))
            return nn.Sequential(*mlp_layers)
             
        self.species_classifier = build_mlp(num_species)
  
        self.breed_classifier = build_mlp(num_breeds)
 
        breed_names = ['american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
                       'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
                       'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
                       'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
                       'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier', 'Abyssinian', 'Bengal',
                       'Birman',
                       'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll',
                       'Russian_Blue',
                       'Siamese', 'Sphynx']
        ## idx to string mappings for the predict method
        self.idx2species = {0: "dog", 1: "cat"}
        self.idx2breed = {idx: breed for idx, breed in enumerate(breed_names)}
 
    def forward(self, x):
        enc_ftrs, x = self.encoder(x)
         
        x_flattened = torch.flatten(x, 1)
        species_pred = self.species_classifier(x_flattened)
        breed_pred = self.breed_classifier(x_flattened)
         
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        mask_pred = self.head(out)
             
        return species_pred, breed_pred, mask_pred
         
    def predict(self, image):
        """
        Receives an image and returns predictions
        input: image (torch.Tensor) - C x H x W
        output: species_pred (string), breed_pred (string), mask_pred (torch.Tensor) - H x W
        """
  
        # Compute the predictions for the input image
        image = image.unsqueeze(0)
        species_pred, breed_pred, mask_pred = self.forward(image)
        # Turn the probabilities into prediction strings
        species_pred_class = self.idx2species[torch.argmax(species_pred, dim=1).item()]
        top3_breed_indices = torch.topk(breed_pred, k=3, dim=1).indices.squeeze(0)
        breed_pred_classes = tuple(self.idx2breed[idx.item()] for idx in top3_breed_indices)
        mask_pred = torch.argmax(mask_pred, dim=1).squeeze(0)
  
        return species_pred_class, breed_pred_classes, mask_pred
  
  
def compute_seg_metrics(pred, target, class_idx):
    pred = pred == class_idx
    target = target == class_idx
    TP = (pred & target).sum()
    FP = (pred & ~target).sum()
    FN = (~pred & target).sum()
  
    iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall, iou
  
def evaluate(model, dataloader):
    model.eval()
    metrics = {"iou": 0, "min_iou": 0, "precision": 0, "recall": 0}
    num_batches = 0
    species_acc, breed_acc, breed_top3_acc = 0, 0, 0
    with torch.no_grad():
        for images, species_labels, breed_labels, masks in dataloader:
  
            images, species_labels, breed_labels, masks = (
                images.to(device), species_labels.to(device), breed_labels.to(device), masks.to(device)
            )
            species_pred, breed_pred, mask_pred = model(images)
            
            species_acc += (torch.sum(torch.argmax(species_pred, dim=1) == species_labels))/len(species_labels)
            breed_acc += (torch.sum(torch.argmax(breed_pred, dim=1) == breed_labels))/len(breed_labels)
            breed_top3 = torch.topk(breed_pred, k=3, dim=1).indices
            breed_top3_acc += torch.sum(torch.any(breed_top3 == breed_labels.unsqueeze(1), dim=1))/len(breed_labels)
              
  
            mask_pred = torch.argmax(mask_pred, dim=1)
            min_iou = 10.0
            for class_idx in range(3):
                precision, recall, iou = compute_seg_metrics(mask_pred, masks, class_idx)
                if iou < min_iou:
                    min_iou = iou
                metrics["iou"] += iou
                metrics["precision"] += precision
                metrics["recall"] += recall
            metrics["min_iou"] += min_iou
  
            num_batches+=1
        metrics = {k:v/(3 * num_batches) for k,v in metrics.items()}
        metrics["min_iou"] = metrics["min_iou"] * 3
        species_acc/=num_batches
        breed_acc/=num_batches
        breed_top3_acc/=num_batches
    print(f"Species accuracy:{species_acc}, breed accuracy:{breed_acc}, top 3 breed accuracy:{breed_top3_acc}\n mask metrics:{metrics}")
    return torch.argmax(species_pred, dim=1), mask_pred
  
def train(model, train_loader, valid_loader, optimizer, criterion, species_criterion, mask_criterion, epoch, scheduler=None):
  
    for e in range(epoch):
        train_loss = 0.0
        bar = tqdm(train_loader)
        bar.set_description(f"epoch={e}")
        model.train()
        for images, species_labels, breed_labels, masks in bar:
            images, species_labels, breed_labels, masks = (
                images.to(device), species_labels.to(device), breed_labels.to(device), masks.to(device)
            )
            optimizer.zero_grad()
            species_pred, breed_pred, mask_pred = model(images)
            loss_species = species_criterion(species_pred, species_labels)
            loss_breed = criterion(breed_pred, breed_labels)
            loss_mask = mask_criterion(mask_pred, masks)
  
            loss = loss_species + loss_breed + loss_mask
            loss.backward()
            optimizer.step()
            bar.set_postfix({"total ss": loss.item(), "breed_loss": loss_breed.item(), "species_loss": loss_species.item(), "masks_loss": loss_mask.item()})
            train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
  
        #print("\nTraining Model evaluation:")
        #evaluate(model, train_loader)
        print("\nValidation:")
        evaluate(model, valid_loader)
  
  
if __name__ == "__main__":
    pass