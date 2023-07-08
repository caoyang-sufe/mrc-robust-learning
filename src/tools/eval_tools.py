# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import math
import numpy
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


# @param y_prob: [torch.FloatTensor], (batch_size, n_option)
# @param y_true: [torch.LongTensor], (batch_size, )
# Evaluate single-choice outputs
def evaluate_single_choice(y_prob, y_true):
    y_pred = torch.max(y_prob, dim=-1)[1]
    score = torch.sum((y_pred == y_true).long())
    return {"accuracy": torch.true_divide(score, y_true.size(0)).item(),
            "y_pred": y_pred,
            "score": score.item(),
            }

# @param model_name                 : Str
# @param train_logging_dataframe	: pandas.DataFrame['epoch', 'iteration', 'loss', 'accuracy']
# @param dev_logging_dataframe		: [pandas.DataFrame]('epoch' 'accuracy')
# @param train_
def plot_single_choice(model_name,
                       train_logging_dataframe,
                       dev_logging_dataframe,
                       train_plot_export_path,
                       dev_plot_export_path,
                       ):
    if train_logging_dataframe is not None:
        plt.rcParams["figure.figsize"] = (12., 9.)
        epochs = train_logging_dataframe["epoch"]
        losses = train_logging_dataframe["loss"]
        accuracies = train_logging_dataframe["accuracy"]
        means = train_logging_dataframe.groupby("epoch").mean()
        plt.plot(means.index, means["loss"], label='loss')
        plt.plot(means.index, means["accuracy"], label='accuracy')
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title(f"Train Plot for {model_name}")
        plt.legend()
        plt.show()

        xs = numpy.linspace(0, epochs.max(), len(epochs))
        figure = plt.figure(1)
        host = HostAxes(figure, [0.15, 0.1, .65, 0.8])
        par1 = ParasiteAxes(host, sharex=host)
        host.parasites.append(par1)
        host.set_ylabel('loss')
        host.set_xlabel('epoch')
        host.axis['right'].set_visible(False)
        par1.axis['right'].set_visible(True)
        par1.set_ylabel('accuracy')
        par1.axis['right'].major_ticklabels.set_visible(True)
        par1.axis['right'].label.set_visible(True)
        figure.add_axes(host)
        p1, = host.plot(xs, losses, label='loss')
        p2, = par1.plot(xs, accuracies, label='accuracy')
        par1.set_ylim(0, 1.2)
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        host.set_xticks(list(range(max(epochs) + 1)))
        host.legend()
        plt.title(f'Train Plot for {model_name}')
        if train_plot_export_path is None:
            plt.show()
        else:
            plt.savefig(train_plot_export_path)
            plt.close()
    if dev_logging_dataframe is not None:
        epochs = dev_logging_dataframe['epoch']
        accuracies = dev_logging_dataframe['accuracy']
        plt.plot(epochs, accuracies, label='accuracy')
        plt.xlabel('epoch')
        plt.ylabel('metric')
        plt.title(f'Valid Plot for {model_name}')
        plt.legend()
        if dev_plot_export_path is None:
            plt.show()
        else:
            plt.savefig(dev_plot_export_path)
            plt.close()
