
filename_txt  = '/home/yufei/Documents/transplan-master (1)/GX010069.csv'

from csv import reader
import cv2

path = '/home/yufei/Documents/transplan-master (1)/GX010069/'
save_path = '/home/yufei/Documents/transplan-master (1)/GX010069/'

area =[]
i = 0
with open(filename_txt, 'r') as f:

    csv_reader = reader(f)
    # header = next(csv_reader)
    # # Check file as empty
    # if header != None:
    #     # Iterate over each row after the header in the csv
    for row in csv_reader:
        print(row)
        img = cv2.imread(path + str(int(row[0])+1).zfill(3) + '.jpg')

        # img = img.resize((2704,1502))
        # img = cv2.resize(img, (2704,1502), interpolation=cv2.INTER_AREA)
        x1 = int(float(row[2]))
        y1 = int(float(row[3]))

        x2 = int(float(row[2]) + float(row[4]))
        y2 = int(float(row[3]) + float(row[5]))

        pts1 = (x1, y1)
        pts2 = (x2, y2)
        i += 1
        if int(row[0]) > 1000:
            break
        # if int(float(row[4])* float(row[5])) > 100000:
        #     continue
        if row[1] == 'car' or row[1] == 'truck' :
            area.append(int(float(row[4])* float(row[5])))

        if row[1] == 'car' or row[1] == 'truck' or row[1] == 'bus' :
            cv2.rectangle(img, pts1, pts2,(0, 255, 0), 3)

        #
            cv2.imwrite(path + str(int(row[0])+1).zfill(3) + '.jpg', img)
        #
        # cv2.imshow("lalala", img)
        # k = cv2.waitKey(0)  # 0==wait forever

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# plt.hist(area, bins=30)  # density=False would make counts
# plt.ylabel('Frequency')
# plt.xlabel('Bounding Box Area')
# plt.savefig('/home/yufei/Downloads/GX010069.png')
# plt.show()
