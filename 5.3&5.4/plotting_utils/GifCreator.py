from PIL import Image
import os

# File contains Gif creation and conversion operations

# Hard limit to number of frames that an image can contain
max_frames = 300

# Looks for a folder named steps in a directory, takes all numbered png files out of directory and mashes them into a gif.
def SaveToGif(name, src_dir, dest_dir, extra_frames = 0, repeats = 0, replace_old = False):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if os.path.exists(dest_dir+'{}.gif'.format(name)):
        if (not(replace_old)):
            print('A file already exists with the name {}.gif. GIF creation failed'.format(name))
            return
    if os.path.exists(src_dir):
        parentDir = src_dir
        frames = []
        framecount = 0
        if not os.path.exists(parentDir+'/00000.png'):
            print('Error occured while creating gif at:', src_dir,'\nERROR: STARTING FRAME NOT FOUND')
            return
        for i in range(extra_frames):
            frames.append(Image.open(parentDir+'/00000.png'))
            for jdx in range(1,repeats):
                frames.append(Image.open(parentDir+'/00000.png'))
            framecount+=1
        for framenum in range(1,max_frames + 1):
            if os.path.exists(parentDir+f'/{framenum:05d}.png'):
                frames.append(Image.open(parentDir+f'/{framenum:05d}.png'))
                for jdx in range(1,repeats):
                    frames.append(Image.open(parentDir+f'/{framenum:05d}.png'))
            else:
                for i in range(extra_frames - 1):
                    frames.append(Image.open(parentDir+f'/{(framenum-1):05d}.png'))
                    for jdx in range(1,repeats):
                        frames.append(Image.open(parentDir+f'/{(framenum-1):05d}.png'))
                break
        # Save into a GIF file that loops forever
        frames[0].save(dest_dir+'/{}.gif'.format(name), format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
    else:
        print('Error occured while creating gif at:', src_dir,'\nERROR: SOURCE FOLDER NOT FOUND')

