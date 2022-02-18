result_textfile_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/detection_result_detectron2.txt"
modified_textfile_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/detection_result_detectron2_modified.txt"
num_header_lines = 4
desired_classes = [2, 7]
#  car: 2 truck:7
from tqdm import tqdm

def pars_line(line):
    splits = line.split(" ")
    # print(splits)
    return int(splits[0]), int(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])

lines =  None
with open(result_textfile_path, "r") as fp:
    lines = fp.readlines()
# input format is like "frame# class# score x1 y1 x2 y2 " for each line 

with open(modified_textfile_path, "w") as fp:
    for line in tqdm(lines[num_header_lines:]):
        fn, clss, score, x1, y1, x2, y2 = pars_line(line)
        if clss in desired_classes:
            fp.write(f"{fn+1} {clss} {x1} {y1} {x2} {y2} {score}\n")
# The output should be of the format - 'fname','v','x','y','w','h','c' in each line.