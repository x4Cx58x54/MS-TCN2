# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from configs import configs
from utils import read_nonempty_lines

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main(config, overlap=[.1, .25, .5, .75, .9]):

    gt_path = config.gt_path
    recog_path = config.results_dir

    file_list = config.vid_test_list_file

    mapping_file = config.mapping_file

    actions = read_nonempty_lines(mapping_file)
    action_ids = []
    action_names = []
    for a in actions:
        action_ids.append(a.split(' ')[0])
        action_names.append(a.split(' ')[1])

    list_of_videos = read_nonempty_lines(file_list)

    tp, fp, fn = np.zeros(len(overlap)), np.zeros(len(overlap)), np.zeros(len(overlap))

    correct = 0
    total = 0
    edit = 0

    gt_content_total = []
    recog_content_total = []

    for vid in list_of_videos:
        gt_content = read_nonempty_lines(os.path.join(gt_path, vid))
        recog_content = read_nonempty_lines(os.path.join(recog_path, os.path.splitext(vid)[0]))

        if abs(len(gt_content)-len(recog_content)) > 3:
            print('WARNING: inconsistent gt and pred lengths')

        # ml = min(len(recog_content), len(gt_content))
        # gt_content = gt_content[:ml]
        # recog_content = recog_content[:ml]

        gt_content = gt_content[:len(recog_content)]


        gt_content_total.extend(gt_content)
        recog_content_total.extend(recog_content)

        assert len(gt_content) == len(recog_content)

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    # confusion matrix
    cm = confusion_matrix(gt_content_total, recog_content_total, labels=action_names)
    cm_col_normalized = cm.astype('float') #/ cm.sum(axis=1)

    # standard
    cm_disp = ConfusionMatrixDisplay(cm_col_normalized)
    # plt.rcParams["figure.figsize"] = (20, 20)
    plt.rcParams["figure.figsize"] = (40, 40)
    cm_disp.plot(colorbar=False)
    # cm_disp.plot(values_format='.2f', cmap=plt.cm.viridis)

    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.matshow(cm_col_normalized, cmap=plt.cm.viridis, alpha=0.5)
    # for m in range(cm_col_normalized.shape[0]):
    #     for n in range(cm_col_normalized.shape[1]):
    #         ax.text(x=m,y=n,s=f'{cm_col_normalized[m, n]:.2f}'[1:], va='center', ha='center')

    # ax.set_xlabel('Predictions')
    # ax.set_ylabel('Truths')
    # ax.set_title('Confusion Matrix, normorlized as recalls')
    cm_save_path = os.path.join(recog_path, 'cm.png')
    plt.savefig(cm_save_path, bbox_inches='tight')

    metrics = dict()

    acc = (100*float(correct)/total)
    edit = ((1.0*edit)/len(list_of_videos))
    metrics['Accuracy% / Test'] = acc
    metrics['Edit / Test'] = edit
    for s in range(len(overlap)):
        # avoid zero division
        precision, recall = 0, 0
        if float(tp[s]+fp[s]) > 1e-4 and float(tp[s]+fn[s]) > 1e-4:
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
        if (precision+recall) > 1e-4:
            f1 = 2.0 * (precision*recall) / (precision+recall)
            f1 = np.nan_to_num(f1)*100
            metrics[f'F1% / @{overlap[s]:0.2f}, Test'] = f1
            metrics[f'Precision% / @{overlap[s]:0.2f}, Test'] = precision * 100
            metrics[f'Recall% / @{overlap[s]:0.2f}, Test'] = recall * 100
    for k, v in sorted(metrics.items()):
        print(f'{k:<25}: {v:.4f}')

    return cm_save_path, metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    args = parser.parse_args()

    config = configs[args.config_name]

    main(config)
