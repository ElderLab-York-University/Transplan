initial_textfile_path = "./../Results/GX010069_detections_detectron2.txt"
modified_textfile_path = "./../Results/GX010069_detections_detectron2_modified.txt"
num_header_lines = 0
#  cars: 2 trucks:7
desired_classes = [2, 7]
# input format is like "frame# class# score x1 y1 x2 y2 " for each line 
# The output should be of the format - 'fname','v','x','y','w','h','c' in each line.
from tqdm import tqdm

# a function to parse files using space
def pars_line(line):
    splits = line.split(" ")
    # print(splits)
    return int(splits[0]), int(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])

# read lines from the initial file
lines =  None
with open(initial_textfile_path, "r") as fp:
    lines = fp.readlines()

# write lines to the  modified text file with a the new format
with open(modified_textfile_path, "w") as fp:
    for line in tqdm(lines[num_header_lines:]):
        fn, clss, score, x1, y1, x2, y2 = pars_line(line)
        if clss in desired_classes:
            fp.write(f"{fn+1} {clss} {x1} {y1} {x2} {y2} {score}\n")