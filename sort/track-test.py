from sort import *
from tqdm import tqdm

detection_file = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/detection_result_detectron2_modified.txt"
output_file = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/trackinig_result.txt"


total_time = 0.0
total_frames = 0



mot_tracker = Sort() #create instance of the SORT tracker
seq_dets = np.loadtxt(detection_file, delimiter=' ')

with open(output_file,'w') as out_file:

    for frame in tqdm(range(int(seq_dets[:,0].max()))):
        total_frames += 1
        dets = seq_dets[seq_dets[:, 0]==(frame+1), 2:7]
        # dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame+1,d[4],d[0],d[1],d[2],d[3]),file=out_file)

print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))