from pathlib import Path
import os
import shutil
def upper_quality():
    data_path = Path('./Images')

    root = (data_path.parent / 'edited')
    root.mkdir(exist_ok=True, parents=True)

    for breed in data_path.iterdir():
        print(breed)
        (root / breed.name).mkdir(exist_ok=True, parents=True)
        imgs = list(breed.glob('./*.jpg'))
        
        
        sorted_files = sorted(imgs, key = os.path.getsize)
        length = len(imgs)

        selected = sorted_files[-10:] if 15 < length else sorted_files

        [shutil.copy(str(i.resolve()), str((root / breed.name).resolve())) for i in selected]

def all2one(parent_dir, output_dir):
    p = Path(parent_dir)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for img in p.glob('**/*.jpg'):
        new_name =  Path(output_dir) / img.name
        shutil.copy(img, new_name)

def parse_converted(temp_path='/home/user/hamid/vafa/compression/code/result/jpeg/png'):
    p = Path(temp_path)
    for img in p.glob('**/*.*'):
        dir_name = (img.name.split('_')[0])
        
parse_converted()
#all2one('/home/user/hamid/vafa/compression/Data/DogBreed/edited', '/home/user/hamid/vafa/compression/Data/temp')