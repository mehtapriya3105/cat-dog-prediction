import src.import_data as import_data 
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader 

def get_test_train_data(test_dir , train_dir , BATCH_SIZE = 32 , NUM_WORKERS  = 2):
    data_transforms = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.Resize(size = (64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),

        ])
    test_data_transforms = transforms.Compose([
        
        transforms.Resize(size = (64,64)),
        transforms.ToTensor()
        ])

    train_data= datasets.ImageFolder(root= train_dir,
                                 transform = data_transforms,
                                 target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform = test_data_transforms)

    train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              num_workers = NUM_WORKERS)

    test_dataloader = DataLoader(dataset = test_data,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)

    class_names = train_data.classes

    return train_dataloader , test_dataloader , class_names


def get_data(what:str):
    test_dir , train_dir = import_data.import_data(test_dir , train_dir )
    train_dataloader , test_dataloader , class_names = get_test_train_data()
    if(what == "all"):
        return train_dataloader , test_dataloader , class_names 
    elif(what == "train"):
        return train_dataloader
    elif(what == "test"):
        return test_dataloader
    elif(what == "classes"):
        return class_names
    

