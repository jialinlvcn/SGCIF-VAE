import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import yaml

def load_dataset_config(config_path='./datasets.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset_info(root, data_type=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'datasets.yaml')
    DATASET_CONFIG = load_dataset_config(config_path)

    for name, info in DATASET_CONFIG.items():
        if name == data_type or (data_type is None and os.path.basename(root) == name):
            return info
    raise Exception('Unknown dataset type')

class SARFolder(ImageFolder):  
    def __init__(self, root, data_type, is_train=True, transform=None):
        info = get_dataset_info(root, data_type)
        subdir = info['train'] if is_train else info['test']
        root = os.path.join(root, data_type, subdir)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
        
        super(SARFolder, self).__init__(root, transform)
        self.class_map = info['classes']
        self.fixed_samples = [(path, self._get_target(path)) for path, _ in self.samples]
    
    def _get_target(self, path):
        for key in self.class_map:
            if key in path:
                return self.class_map[key]
        raise Exception('Class not found')
        
    def __getitem__(self, index: int) :
        path, target = self.fixed_samples[index]
        image = Image.open(path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path
    
    def __len__(self):
        return super().__len__()
    
if __name__ == "__main__":
    dataset = SARFolder('./sar_datasets', 'SAR_ACD', is_train=False)
    print(len(dataset))
    for img, label, path in dataset:
        print(img.shape, label, path)
