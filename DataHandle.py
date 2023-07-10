from pathlib import Path
import os
import shutil
def upper_quality(input_path, output_path, top_quality):
    """
    get "top_quality" number of images within a directory 

    top_quality: number of highest quality images to return

    input_path : path to database

    output_pass : path to save highest quality images

    """
    data_path = Path(input_path)

    root = (output_path)
    root.mkdir(exist_ok=True, parents=True)

    for breed in data_path.iterdir():
        print(breed)
        (root / breed.name).mkdir(exist_ok=True, parents=True)
        imgs = list(breed.glob('./*.jpg'))
        
        
        sorted_files = sorted(imgs, key = os.path.getsize)
        length = len(imgs)

        selected = sorted_files[-top_quality:] if top_quality < length else sorted_files

        [shutil.copy(str(i.resolve()), str((root / breed.name).resolve())) for i in selected]

def all2one(parent_dir, output_dir):
    """
    aggregate all images in subdirectories into single directory

    """
    p = Path(parent_dir)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for img in p.glob('**/*.jpg'):
        new_name =  Path(output_dir) / img.name
        shutil.copy(img, new_name)

def parse_converted(temp_path='/home/user/hamid/vafa/compression/code/result/jpeg/png', output_dir=''):
    """
    This funtion moves distorted images from output result of JPEG / HEVC / VVC to  the destination path and their classes
    
    temp_path : Dirctory which the results obtained from JPEG / HEVC / VVC exists there
    output_dir : Directory to move all images from temp_path to their corresponding class directories

    """
    
    p = Path(temp_path)


    for img in p.glob('**/*.*'):
        name = img.name
        cls = img.name.split('_')[0]
        (Path(p) / cls).mkdir(exist_ok=True, parents=True) # create class directory

        shutil.move(img, (Path(p) / cls / name)) # move distorted image to class directories
    p.rmdir()




parse_converted()
#all2one('/home/user/hamid/vafa/compression/Data/DogBreed/edited', '/home/user/hamid/vafa/compression/Data/temp')


