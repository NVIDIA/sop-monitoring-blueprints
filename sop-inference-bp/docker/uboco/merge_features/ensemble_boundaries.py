import subprocess
import os
import ast
import argparse
import json

def iou(interval_1, interval_2):
    """
    interval: list (2 float elements)
    """
    eps = 1e-8 # to avoid zero division
    (s_1, e_1) = interval_1[:2]
    (s_2, e_2) = interval_2[:2]
    # print(s_1)
    # print(e_1)

    # try:
    intersection = max(0., min(e_1, e_2) - max(s_1, s_2))
    # except:
        # pdb.set_trace()
    union = min(max(e_1, e_2) - min(s_1, s_2), e_1 - s_1 + e_2 - s_2)
    iou = intersection / (union + eps)
    return iou

def mean_iou(times, gt):
    total = 0
    for interval in times:
        best_j = -1
        best_iou = -1
        for j in range(len(gt)):
            cur_iou = iou(interval, gt[j])
            if cur_iou > best_iou:
                best_j = j
                best_iou =cur_iou
        total += best_iou
    return total / len(times)

# offset 0 results: 0.529, 0.536
def k_means_method(results, gt, args): # kmeans of all offsets from 0 to 0.9 0.5324, 0.527
    from sklearn.cluster import KMeans
    import numpy as np
    video_data = {}
    video_ids =[x.strip() for x in open(args.ids,"r").readlines()]

    total_p=0
    total_r=0
    for idx, id in enumerate(video_ids):
        boundaries = []
        for i in range(10):
            boundaries.extend([x[0] for x in results[i][idx]] + [results[i][idx][-1][1]])
        boundaries.sort()
        points = np.array(boundaries).reshape(-1,1)
        kmeans = KMeans(n_clusters= round(sum(len(results[i][idx]) for i in range(10) )/10) +1).fit(points)
        centers = kmeans.cluster_centers_.flatten().round(3)
        centers.sort()
        print("CETNERS")
        print(centers)
        labels=kmeans.labels_
        # print("Cluster centers:", centers)
        import matplotlib.pyplot as plt
        plt.scatter(points, np.zeros_like(points), c=labels, cmap='viridis', marker='o')
        plt.scatter(centers, np.zeros_like(centers), c='red', marker='x')
        plt.xlabel("Number Line")
        plt.yticks([])
        plt.title("K-means Clustering on Number Line")
        plt.show()

        centers[0] = 0
        centers[-1] = results[0][idx][-1][-1]
        actual_boundaries = centers

        times = []
        cur_times = []
        for i in range(len(actual_boundaries)-1):
            times.append(actual_boundaries[i])
            times.append(actual_boundaries[i+1])
            cur_times.append(actual_boundaries[i:i+2])
        
        vid_iou_p = mean_iou(cur_times, gt[idx])
        vid_iou_r = mean_iou( gt[idx], cur_times)
        print(f"{id} miou: {vid_iou_p}, {vid_iou_r}")
        total_p += vid_iou_p
        total_r += vid_iou_r

        video_data[os.path.join(args.folder,video_ids[idx]+".mp4")] = {
            "url":"", 
            "time":times,
        }
    print(f"miou: {total_p / 40},{total_r/40}")
    with open(args.output,"w") as f:
        json.dump(video_data,f)

def merge_timestamps(results, gt, args): # merges the timestamps from 0, 0.1, 0.2 offset. improved precision, worse recall: 0.551, 0.520
    print("starting new strategy ...")
    video_data = {}
    video_ids =[x.strip() for x in open(args.ids,"r").readlines()]


    total_p=0
    total_r=0
    for idx, id in enumerate(video_ids):
        boundaries = []
        for i in range(3):
            boundaries.extend([x[0] for x in results[i][idx]] + [results[i][idx][-1][1]])
        boundaries.sort()

        # take timestamps from 0, 0.1, 0.2 offsets, and for each timestamp, if there's at least 2 within 0.3 of each other, that
        # timestamp is counted as a boundary
        actual_boundaries = []
        j = 0
        THRESH = 0.5
        while j < len(boundaries)-1:
            if j < len(boundaries)-2 and boundaries[j+2] - boundaries[j] < THRESH:
                actual_boundaries.append(boundaries[j])
                j += 3
            elif boundaries[j+1]-boundaries[j] < THRESH:
                actual_boundaries.append(boundaries[j])
                j += 2
            else:
                j += 1
        times = []
        cur_times = []
        for i in range(len(actual_boundaries)-1):
            times.append(actual_boundaries[i])
            times.append(actual_boundaries[i+1])
            cur_times.append(actual_boundaries[i:i+2])
        
        vid_iou_p = mean_iou(cur_times, gt[idx])
        vid_iou_r = mean_iou( gt[idx], cur_times)
        print(f"{id} miou: {vid_iou_p}, {vid_iou_r}")
        total_p += vid_iou_p
        total_r += vid_iou_r

        video_data[os.path.join(args.folder,video_ids[idx]+".mp4")] = {
            "url":"", 
            "time":times,
        }
    
    
    print(f"miou: {total_p / 40},{total_r/40}")
    with open(args.output,"w") as f:
        json.dump(video_data,f)

# offset is 0, 1, .. 9 representing 0.0, 0.1, ... 0.9 offset
def get_results_for_offset(offset, results, gt, args): # merges the timestamps from 0, 0.1, 0.2 offset. improved precision, worse recall: 0.551, 0.520
    total_p, total_r = 0,0
    video_data = {}
    video_ids =[x.strip() for x in open(args.ids,"r").readlines()]

    print( f"results for offset 0.{offset}")
    for idx, id in enumerate(video_ids):
        
        vid_iou_p = mean_iou( results[offset][idx], gt[idx])
        vid_iou_r = mean_iou(  gt[idx] , results[offset][idx])
        print(f"{id} miou: {vid_iou_p}, {vid_iou_r}")
        total_p += vid_iou_p
        total_r += vid_iou_r
        times = []
        for i in range(len(results[offset][idx])):
            times.append(results[offset][idx][0])
            times.append(results[offset][idx][1])
        video_data[os.path.join(args.folder,video_ids[idx]+".mp4")] = {
            "url":"", 
            "time":times,
        }
    print(f"miou: {total_p / 40}, {total_r/ 40}")

    with open(args.output,"w") as f:
        json.dump(video_data,f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../dense_captioning_eval/merge_times_intern_times.json")
    parser.add_argument("--folder", type=str, default="/home/alhu/nv-label-studio-video/data/benchmark")
    parser.add_argument("--ids", type=str, default="../dense_captioning_eval/video_ids.txt")
    parser.add_argument("--gt", type=str, default="../dense_captioning_eval/benchmark_gt.txt")

    
    args = parser.parse_args()

    results = []
    for i in range(10):
        result_path = f"results/benchmark_test_intern_cliplen1_start0{i}_complete.txt"
        if not os.path.exists(result_path):
            command=f"python pseudo_event_boundaries_inference.py --v_feat_dirs /home/alhu/nv-label-studio-video/HERO_Video_Feature_Extractor/clip/output/clip-intern_cliplen1_start0{i}_features/ /home/alhu/nv-label-studio-video/HERO_Video_Feature_Extractor/slowfast/output/slowfast_features_cliplen1_start0{i} --result_path results/benchmark_test_intern_cliplen1_start0{i}_complete.txt --eval_path benchmark_test.jsonl --data_path /home/alhu/nv-label-studio-video/data/benchmark --start_time 0.{i}"
            result = subprocess.run(command.split(' '), check=True, text=True, capture_output=True)
            print("Output:", result.stdout)
            print("Error:", result.stderr)
            print("Return Code:", result.returncode)
        
        results.append([ast.literal_eval(x.strip()) for x in open(result_path,"r").readlines() ])
    
    gt = [ast.literal_eval(x.strip()) for x in open(args.gt,"r").readlines() ]
    

    merge_timestamps(results, gt, args) # k_means_method(results, gt, args)
