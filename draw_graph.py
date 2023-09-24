import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from config import performance_path, images_path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def filter_section(x_list, y_list, x_section=(0, 1), y_section=(0, 1), x_thre=0.01, y_thre=0.01):
    x0, x1 = x_section
    y0, y1 = y_section
    max_x = max(x_list)
    max_y = max(y_list)
    i = 0
    while i < len(x_list) - 1:
        x_i, y_i = x_list[i], y_list[i]
        if not (x0 <= x_i <= x1 and y0 <= y_i <= y1):
            i = i + 1
            continue
        j = i + 1
        while j < len(x_list):
            x_j, y_j = x_list[j], y_list[j]
            if not (x0 <= x_j <= x1 and x_j - x_i <= x_thre):
                break
            if y0 <= y_j <= y1 and abs(y_j - y_i) <= y_thre and x_j != max_x and y_j != max_y:
                x_list.pop(j)
                y_list.pop(j)
            else:
                j = j + 1
        i = i + 1

    return x_list, y_list


class DrawScatter(object):
    max_scatter_count = 3
    marker_list = ['o', '^', 'v']
    color_list = ['#6868ff', '#ff8b26', '#ff0000']
    size_list = [15, 25, 35]
    edgecolor_list = ['#555555', '#555555', '#000000']
    zorder_list = [2, 4, 6]

    def __init__(self):
        self.scatter_count = 0
        self.ax = None

    def figure(self, figsize, dpi=300, **kwargs):
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        self.ax = plt.subplot(1, 1, 1)

    def draw(self, x, y, marker=None, c=None, s=None, lw=0.2, ec=None, zorder=None, label=None, **kwargs):
        if self.scatter_count >= self.max_scatter_count:
            self.scatter_count = self.max_scatter_count - 1

        if marker is None:
            marker = self.marker_list[self.scatter_count]
        if c is None:
            c = self.color_list[self.scatter_count]
        if s is None:
            s = self.size_list[self.scatter_count]
        if ec is None:
            ec = self.edgecolor_list[self.scatter_count]
        if zorder is None:
            zorder = self.zorder_list[self.scatter_count]

        plt.scatter(
            x,
            y,
            marker=marker,
            c=c,
            s=s,
            lw=lw,
            ec=ec,
            zorder=zorder,
            label=label,
            **kwargs
        )

        self.scatter_count = self.scatter_count + 1

    def xylabel(self, xlabel, ylabel='benefit', fontsize=28, **kwargs):
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize, **kwargs)

    def xylim(self, xlim=(0, 1), ylim=(0, 1)):
        plt.xlim(xlim)
        plt.ylim(ylim)

    def xyticks(self, xticks, yticks, fontsize=14, **kwargs):
        plt.xticks(xticks, fontsize=fontsize, **kwargs)
        plt.yticks(yticks, fontsize=fontsize, **kwargs)

    def legend(self, loc='best', fontsize=12, **kwargs):
       plt.legend(loc=loc, fontsize=fontsize, **kwargs)

    def grid(self, c='#eeeeee', ls='--', zorder=0, **kwargs):
        plt.grid(True, c=c, ls=ls, zorder=zorder, **kwargs)

    def set_square_shape(self, ax=None):
        if ax is None:
            ax = self.ax
        ax.set_aspect('equal', adjustable='box')

    def adjust_padding(self, top=None, right=None, bottom=None, left=None, **kwargs):
        plt.subplots_adjust(top=top, right=right, bottom=bottom, left=left, **kwargs)

    def annotate(self, text, xy, xytext, arrowprops=None, fontsize=30, fontfamily='Consolas', zorder=12, **kwargs):
        if arrowprops is None:
            arrowprops = {'facecolor': 'green', 'arrowstyle': 'fancy'}
        plt.annotate(
            text=text,
            xy=xy,
            xytext=xytext,
            arrowprops=arrowprops,
            fontsize=fontsize,
            fontfamily=fontfamily,
            zorder=zorder,
            **kwargs
        )

    def save(self, path):
        plt.savefig(path)


class DrawWholeScatter(DrawScatter):
    def __init__(self):
        super(DrawWholeScatter, self).__init__()

    def xyticks(self,
        xticks=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        yticks=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        fontsize=14, **kwargs):
        plt.xticks(xticks, fontsize=fontsize, **kwargs)
        plt.yticks(yticks, fontsize=fontsize, **kwargs)

    @staticmethod
    def draw_magnify_border(x_magnify, y_magnify, c='red', ls='--', lw=2, zorder=14, **kwargs):
        # 修正边界值
        left, right = x_magnify
        bottom, top = y_magnify
        delta = 0.005
        if left <= 0:
            left = left + delta
        if right >= 1:
            right = right - delta
        if bottom <= 0:
            bottom = bottom + delta
        if top >= 1:
            top = top - delta

        # 画边框
        plt.plot(
            [left, right, right, left, left],
            [top, top, bottom, bottom, top],
            c=c,
            ls=ls,
            lw=lw,
            zorder=zorder,
            **kwargs
        )


class DrawMagnifyScatter(DrawScatter):
    size_list = [150, 250, 350]

    def __init__(self, x_magnify, y_magnify):
        super(DrawMagnifyScatter, self).__init__()

        self.left, self.right = x_magnify
        self.bottom, self.top = y_magnify
        if self.right - self.left != self.top - self.bottom:
            print('警告：非正方形放大区域')

    def xylim(self):
        plt.xlim((self.left, self.right))
        plt.ylim((self.bottom, self.top))

    def xyticks(self, xticks=None, yticks=None, fontsize=14, long_x_axis=True, **kwargs):
        if long_x_axis:
            allow_num_ticks = [10, 9, 11, 8, 12, 13]
        else:
            allow_num_ticks = [8, 7, 6, 5, 9, 4, 10]

        if xticks is None:
            x_diff = int(self.right * 100) - int(self.left * 100)
            for num_ticks in allow_num_ticks:
                if x_diff % (num_ticks - 1) == 0:
                    step = 0.01 * x_diff / (num_ticks - 1)
                    xticks = [self.left + i * step for i in range(num_ticks)]
                    break
        if xticks is None:
            print('警告：无法自动设置X轴刻度，请自行设置')
        else:
            plt.xticks(xticks, fontsize=fontsize, **kwargs)

        if yticks is None:
            y_diff = int(self.top * 100) - int(self.bottom * 100)
            yticks = None
            for num_ticks in allow_num_ticks:
                if y_diff % (num_ticks - 1) == 0:
                    step = 0.01 * y_diff / (num_ticks - 1)
                    yticks = [self.bottom + i * step for i in range(num_ticks)]
                    break
        if yticks is None:
            print('警告：无法自动设置Y轴刻度，请自行设置')
        else:
            plt.yticks(yticks, fontsize=fontsize, **kwargs)

    def in_magnify(self, point):
        return self.left <= point[0] <= self.right and self.bottom <= point[1] <= self.top

    def get_best_point(self, x, y):
        in_magnify_data = list(filter(lambda t: self.in_magnify(t), zip(x, y)))
        df = pd.DataFrame(in_magnify_data, columns=['performance', 'benefit'])
        best_var_index = df['performance'].values.argmax()
        best_benefit_index = df['benefit'].values.argmax()
        best_var_x, best_var_y = df.iloc[best_var_index, 0], df.iloc[best_var_index, 1]
        best_benefit_x, best_benefit_y = df.iloc[best_benefit_index, 0], df.iloc[best_benefit_index, 1]

        return (best_var_x, best_var_y), (best_benefit_x, best_benefit_y)

    def annotate_best_point(self, x, y, performance_loc=None, benefit_loc=None, arrowprops=None, fontsize=30, fontfamily='Consolas', zorder=12, **kwargs):
        (best_var_x, best_var_y), (best_benefit_x, best_benefit_y) = self.get_best_point(x, y)
        if performance_loc is None:
            performance_loc = (best_var_x, best_var_y)
        if benefit_loc is None:
            benefit_loc = (best_benefit_x, best_benefit_y)
        self.annotate(
            text='({:.4f},{:.4f})'.format(best_var_x, best_var_y),
            xy=(best_var_x, best_var_y),
            xytext=performance_loc,
            arrowprops=arrowprops,
            fontsize=fontsize,
            fontfamily=fontfamily,
            zorder=zorder,
            **kwargs
        )
        self.annotate(
            text='({:.4f},{:.4f})'.format(best_benefit_x, best_benefit_y),
            xy=(best_benefit_x, best_benefit_y),
            xytext=benefit_loc,
            arrowprops=arrowprops,
            fontsize=fontsize,
            fontfamily=fontfamily,
            zorder=zorder,
            **kwargs
        )


def get_test_data():
    var_name_list = ['sensitivity', 'specificity', 'accuracy', 'auc_nc', 'auc_mci', 'auc_de', 'ap_nc', 'ap_mci', 'ap_de']
    model_name_list = ['MRI', 'nonImg', 'Fusion']
    data_dict = {}
    for model_name in model_name_list:
        data = pd.read_csv(os.path.join(performance_path, '{}_result.csv'.format(model_name)))
        for var_name in var_name_list:
            var_data = data[[var_name, 'benefit']].sort_values(var_name)
            x = var_data[var_name].values
            y = var_data['benefit'].values
            if var_name in data_dict:
                data_dict[var_name][model_name] = (x, y)
            else:
                data_dict[var_name] = {model_name: (x, y)}

    return model_name_list, data_dict


def test_draw_whole_scatter():
    # 载入数据
    draw_config = OrderedDict({
        'sensitivity': '$sensitivity$',
        'specificity': '$specificity$',
        'accuracy': '$accuracy$',
        'auc_nc': '$AUC_{NC}$',
        'auc_mci': '$AUC_{MCI}$',
        'auc_de': '$AUC_{DE}$',
        'ap_nc': '$AP_{NC}$',
        'ap_mci': '$AP_{MCI}$',
        'ap_de': '$AP_{DE}$'
    })
    model_name_list, data_dict = get_test_data()

    # 画图
    filename_list = []
    for var_name, xlabel in draw_config.items():
        draw_obj = DrawWholeScatter()
        draw_obj.figure((5, 5))
        for i, model_name in enumerate(model_name_list):
            x, y = data_dict[var_name][model_name]
            x, y = filter_section(x.tolist(), y.tolist(), x_section=(0, 1), y_section=(0, 1), x_thre=0.01, y_thre=0.01)
            draw_obj.draw(x, y, label=model_name)
        draw_obj.legend()
        draw_obj.xylabel(xlabel)
        draw_obj.xylim()
        draw_obj.xyticks()
        draw_obj.grid()
        draw_obj.set_square_shape()
        draw_obj.adjust_padding(top=0.9, right=0.95, bottom=0.16, left=0.16)
        filename = '{}_whole_scatter.png'.format(var_name)
        draw_obj.save(os.path.join(images_path, filename))
        filename_list.append(filename)

    return filename_list


def test_draw_magnify_scatter():
    # 载入数据
    draw_config = OrderedDict({
        'sensitivity': ('$Sensitivity$', (0.67, 0.87), (0.79, 0.99), 'upper left', (0.74, 0.96), (0.74, 0.96)),
        'specificity': ('$Specificity$', (0.79, 0.95), (0.77, 0.93), 'upper left', (0.835, 0.89), (0.835, 0.915)),
        'accuracy': ('$Accuracy$', (0.72, 0.9), (0.8, 0.98), 'upper left', (0.82, 0.96), (0.775, 0.92)),
        'auc_nc': ('$AUC_{NC}$', (0.84, 1), (0.8, 0.96), 'upper left', (0.93, 0.945), (0.89, 0.925)),
        'auc_mci': ('$AUC_{MCI}$', (0.77, 0.95), (0.78, 0.96), 'upper left', (0.875, 0.945), (0.82, 0.92)),
        'auc_de': ('$AUC_{DE}$', (0.84, 1.0), (0.78, 0.94), 'upper left', (0.89, 0.895), (0.90, 0.925)),
        'ap_nc': ('$AP_{NC}$', (0.78, 0.96), (0.78, 0.96), 'upper left', (0.88, 0.945), (0.83, 0.925)),
        'ap_mci': ('$AP_{MCI}$', (0.8, 0.98), (0.78, 0.96), 'upper left', (0.895, 0.925), (0.855, 0.945)),
        'ap_de': ('$AP_{DE}$', (0.77, 0.95), (0.78, 0.96), 'upper left', (0.875, 0.92), (0.82, 0.945))
    })
    model_name_list, data_dict = get_test_data()

    # 画图
    filename_list = []
    for var_name, (xlabel, x_magnify, y_magnify, legend_loc, performance_loc, benefit_loc) in draw_config.items():
        draw_obj = DrawMagnifyScatter(x_magnify, y_magnify)
        draw_obj.figure((10, 5))
        whole_x, whole_y = [], []
        for model_name in model_name_list:
            x, y = data_dict[var_name][model_name]
            x, y = x.tolist(), y.tolist()
            whole_x.extend(x)
            whole_y.extend(y)
            x, y = filter_section(x, y, x_section=x_magnify, y_section=y_magnify, x_thre=0.005, y_thre=0.005)
            draw_obj.draw(x, y, lw=1, label=model_name)
        draw_obj.xylim()
        draw_obj.xylabel(xlabel)
        draw_obj.xyticks()
        draw_obj.legend(legend_loc, fontsize=16)
        draw_obj.grid()
        draw_obj.adjust_padding(top=0.95, right=0.95, bottom=0.15, left=0.1)
        draw_obj.annotate_best_point(whole_x, whole_y, performance_loc, benefit_loc)
        filename = '{}_magnify_scatter.png'.format(var_name)
        draw_obj.save(os.path.join(images_path, filename))
        filename_list.append(filename)

    return filename_list


def test_draw_subgraph_scatter():
    # 载入数据
    draw_config = OrderedDict({
        'sensitivity': ('$Sensitivity$', (0.67, 0.87), (0.79, 0.99), 'upper left'),
        'specificity': ('$Specificity$', (0.79, 0.95), (0.77, 0.93), 'upper left'),
        'accuracy': ('$Accuracy$', (0.72, 0.9), (0.8, 0.98), 'upper left'),
        'auc_nc': ('$AUC_{NC}$', (0.84, 1), (0.8, 0.96), 'upper left'),
        'auc_mci': ('$AUC_{MCI}$', (0.77, 0.95), (0.78, 0.96), 'upper left'),
        'auc_de': ('$AUC_{DE}$', (0.84, 1.0), (0.78, 0.94), 'upper left'),
        'ap_nc': ('$AP_{NC}$', (0.78, 0.96), (0.78, 0.96), 'upper left'),
        'ap_mci': ('$AP_{MCI}$', (0.8, 0.98), (0.78, 0.96), 'upper left'),
        'ap_de': ('$AP_{DE}$', (0.77, 0.95), (0.78, 0.96), 'upper left')
    })
    model_name_list, data_dict = get_test_data()

    # 画图
    filename_list = []
    for var_name, (xlabel, x_magnify, y_magnify, legend_loc) in draw_config.items():
        draw_whole_obj = DrawWholeScatter()
        draw_magnify_obj = DrawMagnifyScatter(x_magnify, y_magnify)

        plt.figure(figsize=(10, 5), dpi=300)
        left_ax = plt.subplot(1, 2, 1)
        for model_name in model_name_list:
            x, y = data_dict[var_name][model_name]
            draw_whole_obj.draw(x, y, lw=0.2, label=model_name)
        draw_whole_obj.draw_magnify_border(x_magnify, y_magnify)
        draw_whole_obj.xylabel(xlabel, fontsize=20)
        draw_whole_obj.xyticks()
        draw_whole_obj.xylim()
        draw_whole_obj.legend()
        draw_whole_obj.set_square_shape(left_ax)
        draw_whole_obj.grid()
        draw_whole_obj.adjust_padding(left=0.08, bottom=0.15, top=0.9)

        right_ax = plt.subplot(1, 2, 2)
        for model_name in model_name_list:
            x, y = data_dict[var_name][model_name]
            x, y = filter_section(x.tolist(), y.tolist(), x_section=x_magnify, y_section=y_magnify, x_thre=0.005, y_thre=0.005)
            draw_magnify_obj.draw(x, y, lw=1, label=model_name)
        draw_magnify_obj.xylim()
        draw_magnify_obj.xylabel(xlabel, fontsize=20)
        draw_magnify_obj.xyticks(long_x_axis=False)
        draw_magnify_obj.legend(legend_loc, fontsize=16)
        draw_magnify_obj.grid()
        draw_magnify_obj.set_square_shape(right_ax)
        draw_magnify_obj.adjust_padding(right=0.97, bottom=0.15, top=0.9)

        filename = '{}_subgraph_scatter.png'.format(var_name)
        draw_whole_obj.save(os.path.join(images_path, filename))
        filename_list.append(filename)

    return filename_list
