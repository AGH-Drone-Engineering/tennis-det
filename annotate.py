import shutil
import os
from PIL import Image
import yaml
from tqdm import tqdm


class RandomDataset:
    def __getitem__(self, i):
        return f'test.jpg', [[0, 100, 200, 300, 400]]
    
    def __len__(self):
        return 10


def save_yolo_dataset(path: str, *, class_names: list[str], **splits):
    '''
    Save a dataset in the YOLO format.

    Args:
        path: Dataset destination path.
        splits: A dictionary of splits to save.
            Each split is a dataset supporting `__getitem__` and `__len__`.
            A dataset contains `(image_path, targets)` tuples where `targets` is a list of bounding boxes.
            Each bounding box is a tuple `(cls, xmin, ymin, xmax, ymax)` in pixel coordinates.
            `cls` is the class index.
    '''

    shutil.rmtree(path, ignore_errors=True)
    for split, data in splits.items():
        print(f'Saving {split} set')
        os.makedirs(os.path.join(path, split, 'images'))
        os.makedirs(os.path.join(path, split, 'labels'))
        def save_image(i):
            image_in_path, target = data[i]
            with Image.open(image_in_path) as image:
                image_path = os.path.join(path, split, 'images', f'{i}.jpg')
                image.save(image_path)
                with open(os.path.join(path, split, 'labels', f'{i}.txt'), 'w') as f:
                    for t in target:
                        c, xmin, ymin, xmax, ymax = t
                        x, y, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
                        x, y, w, h = x / image.width, y / image.height, w / image.width, h / image.height
                        f.write(f'{c} {x} {y} {w} {h}\n')
        for i in tqdm(range(len(data))):
            save_image(i)
    with open(os.path.join(path, 'data.yaml'), 'w') as f:
        yaml.dump({
            'names': class_names,
            'nc': len(class_names),
            **{split: split for split in splits},
        }, f)


def run():
    save_yolo_dataset(
        path='tennis_dataset',
        class_names=['tennis_ball'],
        train=RandomDataset(),
    )


if __name__ == '__main__':
    run()
