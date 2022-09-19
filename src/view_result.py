from agents import *
from models import *
from utils.World import *
import sys
import pickle

import glob
from PIL import Image

"""
This script is to visualize evaluation result.

usage:
python view_result.py path_to_record_file
"""
# instantiate the class
f = open(sys.argv[1], 'rb')
record = pickle.load(f)
dT = record.dT
exec('robot = ' + record.model + '(' + record.algorithm + '(), dT)');
human = HumanBall3D(MobileAgent(), dT);

w = World(dT, human, robot, record)
base.movie(namePrefix='/Users/jacob/Desktop/Changliu/BIS-master 2/new_movie/movie', duration=900, fps=30)
base.run()

# from PIL import Image
# import glob
#
# # Create the frames
# frames = []
# imgs = glob.glob("*.png")
# for i in imgs:
#     new_frame = Image.open(i)
#     frames.append(new_frame)
#
# # Save into a GIF file that loops forever
# frames[0].save('png_to_gif.gif', format='GIF',
#                append_images=frames[1:],
#                save_all=True,
#                duration=300, loop=0)

# filepaths
# fp_in = "/path/to/image_*.png"
fp_in = '/Users/jacob/Desktop/Changliu/BIS-master 2/new_movie/*.png'
fp_out = "/Users/jacob/Desktop/Changliu/BIS-master 2/new_movie/movie.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

