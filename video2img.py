import os



import os
import shutil

data_dir = "./videos/"

for i in os.listdir(data_dir):
    general_txt = i.split(".")[0]
    if not os.path.exists(f"./vox2_crop_fps25/{general_txt}"):
        os.makedirs(f"./vox2_crop_fps25/{general_txt}/")
    os.system(f'ffmpeg -i ./videos/{i} -vf fps=25 ./vox2_crop_fps25/{general_txt}/%06d.png')
    