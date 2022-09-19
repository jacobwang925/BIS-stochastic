import glob
from PIL import Image


fp_in = '/Users/jacob/Desktop/Changliu/BIS-master 2/new_movie/*.png'
fp_out = "/Users/jacob/Desktop/Changliu/BIS-master 2/new_movie/movie.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)