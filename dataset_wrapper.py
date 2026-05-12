from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, ConcatDataset, random_split
from collections import Counter
import torch

class PetFourClassDataset(Dataset):
    def __init__(self, dataset):
        self.samples = []
        self.class_counts = Counter()

        long_haired_cats = {"maine_coon", "persian", "ragdoll"}
        short_haired_cats = {
            "abyssinian", "bengal", "birman", "bombay", "british_shorthair",
            "egyptian_mau", "russian_blue", "siamese", "sphynx"
        }
        long_haired_dogs = {
            "newfoundland", "pomeranian", "samoyed", "saint_bernard", "leonberger",
            "english_setter", "havanese", "japanese_chin", "keeshond",
            "wheaten_terrier", "yorkshire_terrier", "english_cocker_spaniel", "scottish_terrier"
        }
        short_haired_dogs = {
            "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
            "boxer", "chihuahua", "german_shorthaired", "miniature_pinscher",
            "pug", "shiba_inu", "staffordshire_bull_terrier"
        }

        datasets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]

        for ds in datasets:
            for i in range(len(ds)):
                img, label = ds[i]
                breed_name = ds.classes[label].lower().replace(" ", "_")

                if breed_name in long_haired_cats:
                    new_label = 0
                elif breed_name in short_haired_cats:
                    new_label = 1
                elif breed_name in long_haired_dogs:
                    new_label = 2
                elif breed_name in short_haired_dogs:
                    new_label = 3
                else:
                    continue

                self.samples.append((img, new_label))
                self.class_counts[new_label] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, torch.tensor(label)

def get_pet_datasets(img_width=128, img_height=128, root_path='./data'):
    transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor()
    ])

    train_ds = OxfordIIITPet(root=root_path, download=True, target_types='category', split='trainval', transform=transform)
    test_ds = OxfordIIITPet(root=root_path, download=True, target_types='category', split='test', transform=transform)

    full_dataset = ConcatDataset([train_ds, test_ds])
    wrapped_dataset = PetFourClassDataset(full_dataset)

    total_size = len(wrapped_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(wrapped_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


#usage:
#train_dataset, val_dataset, test_dataset = get_pet_datasets(img_width=224, img_height=224,root_path='./data' )