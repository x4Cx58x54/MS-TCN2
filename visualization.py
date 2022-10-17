import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from configs import configs
from utils import read_nonempty_lines


def main(config, video_name, title_suffix='', fps=5, target_dir='visualizations'):

    target_gt_file = os.path.join(target_dir, video_name+'_gt')
    target_pred_file = os.path.join(target_dir, video_name+'_pred')

    gt_path = config.gt_path
    results_dir = config.results_dir
    actions_dict = config.actions_dict

    gt_file = os.path.join(gt_path, video_name + '.txt')
    pred_file = os.path.join(results_dir, video_name)

    gt = read_nonempty_lines(gt_file)
    pred = read_nonempty_lines(pred_file)

    def list_to_period(frame_list, target_file):
        period_wise = []
        with open(target_file, 'w') as f:
            cur_start_t = 1
            cur_action = frame_list[0]
            for t, frame in enumerate(frame_list):
                cur_t = t+1
                if t == len(frame_list) - 1:
                    period_wise.append((cur_start_t, cur_t, frame))
                    f.write(f'{cur_start_t} {cur_t} {frame}\n')
                elif frame != cur_action:
                    period_wise.append((cur_start_t, cur_t-1, cur_action))
                    f.write(f'{cur_start_t} {cur_t-1} {cur_action}\n')
                    cur_start_t = cur_t
                    cur_action = frame
        return period_wise

    gt_period_wise = list_to_period(gt, target_gt_file)
    pred_period_wise = list_to_period(pred, target_pred_file)

    random.seed(0)
    colors = [f'#{random.randint(0, 0xFFFFFF):06X}' for _ in range(len(actions_dict))]
    colors[-1] = '#FFFFFF'

    height = 1
    fig, ax = plt.subplots(figsize=(20,3))

    def plot_timeline(ax, period_wise, typ):
        for a in period_wise:
            start, end, action_name = a
            label = actions_dict[action_name]
            ax.barh(typ, left=start/fps, width=(end-start+1)/fps, label=label, color=colors[label])

    plot_timeline(ax, gt_period_wise, 'gt')
    plot_timeline(ax, pred_period_wise, 'pred')
    plt.title(f'Timeline for {video_name}{title_suffix}')


    patches = [mpatches.Patch(color=colors[x], label=x) for x in actions_dict.values()]
    plt.legend(handles=patches, ncol=17, loc='upper left', bbox_to_anchor=(0,-0.1))
    plt.tight_layout()
    # plt.savefig(os.path.join(target_dir, video_name+'_timeline.png'), bbox_inches='tight')
    vis_fig_save_name = os.path.join(target_dir, 'timeline.png')
    plt.savefig(vis_fig_save_name, bbox_inches='tight')
    return vis_fig_save_name


if __name__ == '__main__':

    config_name = 'ktd_5fps_3072'
    config = configs[config_name]

    video_name = '048_0'

    main(config, video_name)
