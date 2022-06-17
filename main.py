from pathlib import Path

from HGRecognition import CreateDataset

if __name__ == '__main__':
    db = CreateDataset(gesture_name='anything',
                       dst_dir=Path(__file__).resolve().parent / 'dataset',
                       image_suffix='Hello')
    db.createImages()
