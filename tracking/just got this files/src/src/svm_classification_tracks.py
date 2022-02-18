import pandas as pd
import scipy.io
import numpy as np
import cv2
import csv
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colormap = cm.get_cmap('rainbow', 50)
reducedmap = ListedColormap(colormap(np.linspace(1.0, 0.0, 13)))

base_folder = '/media/poorna/D_drive/projects/trans-plan/05March2021/'
intersection_folder = 'Derry Road W at RR25(Bronte Road) (Milton)/'
frame = cv2.imread(base_folder+intersection_folder+'/Homography/drawnworld.png')
gen_model_img = cv2.imread(base_folder+'generalized_model.png')
labelled_tracks_mat = scipy.io.loadmat(base_folder+intersection_folder+'labelled_trajectories.mat')['labelled_trajectories'][0]

X = []
Xfp = []
y = []

for i in range(len(labelled_tracks_mat[0][0])):
    trajectory = labelled_tracks_mat[0][0][i]
    label = labelled_tracks_mat[2][0][i]
    midpt_index = -1
    cumulative_dist = 0
    for i in range(1, len(trajectory)-1):
        cumulative_dist = cumulative_dist + np.linalg.norm(np.array(trajectory[i+1])-np.array(trajectory[i]))
    half_dist = cumulative_dist/2
    cumulative_half_dist = 0
    for i in range(1, len(trajectory)-1):
        cumulative_half_dist = cumulative_half_dist + np.linalg.norm(np.array(trajectory[i+1])-np.array(trajectory[i]))
        if cumulative_half_dist > half_dist:
            midpt_index = i
            break
    #print(midpt_index)
    #print(len(trajectory)/2)
    #print('next')
    if label == 0:
        continue
    Xfp.append([trajectory[0][0], trajectory[0][1], trajectory[midpt_index][0], trajectory[midpt_index][1], trajectory[-1][0], trajectory[-1][1]])
    X.append(trajectory)
    y.append(label)

colors = ['b','g','r','k','c','m','y','fuchsia','brown','orange','crimson','chocolate','steelblue']
tracks_count = 0
for j in range(1,12):
    print(j)
    for track_num,track in enumerate(X):
        if j!=y[track_num]:
            continue
        plt.imshow(frame)
        colorindex = y[track_num]
        if colorindex == 0:
            continue
        #if y[track_num] == 9:
        tracks_count+=1
        #if tracks_count > 25:
        #    break
        color = colors[colorindex]
        x_ = [item[0] for item in track]
        y_ = [item[1] for item in track]
        plt.scatter(x_, y_, c=color, s=20)
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(Xfp, y, test_size=0.25)
classes_train, classes_counts_train = np.unique(y_train, return_counts=True)
classes_test, classes_counts_test = np.unique(y_test, return_counts=True)
maxy = max(y)
miny = min(y)
classes_labels = ['Unknown', 'South right', 'South left', 'South through', 'North right', 'North left', 'North through', 'East right', 'East left', 'East through', 'West right', 'West left', 'West through']
curr_class_labels = []
for i in range(len(classes_test)):
    curr_class_labels.append(classes_labels[classes_test[i]])

single_tracks1 = []
single_tracks2 = []

classes_nums = np.arange(0,12)
print('###### SVM with 3 points ######')
svc_pipeline = make_pipeline(StandardScaler(), SVC(gamma='auto',max_iter=10000))
cross_val_scores = cross_val_score(svc_pipeline, Xfp, y, scoring="accuracy")#, scoring="neg_mean_squared_error", cv=10)
#cross_val_scores_rmse = np.sqrt(cross_val_scores)
print("Mean:", cross_val_scores.mean())
print("Std:", cross_val_scores.std())
clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto',max_iter=100000)).fit(X_train, y_train)
score_svm = clf_svm.score(X_train, y_train)
score_test_svm = clf_svm.score(X_test, y_test)
y_hash_svm = clf_svm.predict(X_test)
cm_svm = metrics.confusion_matrix(y_test, y_hash_svm)
print("Training accuracy - ")
print(score_svm)
print("Test accuracy - ")
print(score_test_svm)
print("Confusion matrix SVM with 2 end points and mid point - ")
print(cm_svm)
print(metrics.classification_report(y_test, y_hash_svm, digits=3))
temp = 0
disp = plot_confusion_matrix(clf_svm, X_test, y_test,
                                 display_labels=curr_class_labels,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
disp.ax_.set_title("Normalized confusion matrix")
plt.setp(disp.ax_.get_xticklabels(), rotation=30, horizontalalignment='right')
print(disp.confusion_matrix)

plt.show()


# for i, y_ in enumerate(y_test):
#     plt.imshow(gen_model_img)
#     if y_ == 4 and y_hash_svm[i] == 10:
#         single_tracks1.append(X[i])
#         xp, yp = zip(*single_tracks1[0])
#         plt.scatter(xp, yp)
#         plt.show()
    # if y_ == 10:
    #     single_tracks2.append(X[i])
    #     xp, yp = zip(*single_tracks2[0])
    #     plt.scatter(xp, yp)




# fig, ax = plt.subplots()
# ax.matshow(norm_cm, cmap=plt.cm.gray)
# cur_labels = [classes_labels[i+1] for i in classes_test]
# xticks = ax.get_xticks()
# #ax.set_xticks(np.arange(0,len(classes_test))-0.5)
# #ax.set_yticks(np.arange(0,len(classes_test))-0.5)
# ax.set_xticklabels(['']+cur_labels)
# ax.set_yticklabels(['']+cur_labels)
#plt.show()
# Night time score and confusion matrix
# score_night_svm = clf_svm.score(X_night_fp, y_night)
# y_hash_night_svm = clf_svm.predict(X_night_fp)
# cm_night_svm = metrics.confusion_matrix(y_night, y_hash_night_svm)
# print(metrics.classification_report(y_night, y_hash_night_svm, digits=3))
# print("Night Test accuracy - ")
# print(score_night_svm)
# print("Night Confusion matrix SVM with 2 end points and mid point - ")
# print(cm_night_svm)


