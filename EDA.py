# Samuel Moijueh
############################
## This script generates the Exploratory Data Analysis plots
############################

""" Visualization """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import nbimporter
from IPython.display import display, HTML

import librosa.display
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pydot
pydot.Dot.create(pydot.Dot())

SAMPLE_RATE = 22050
fg_color = 'white'

# display(HTML("""
# <style>
# .output {
#     display: table-cell;
#     text-align: center;
#     vertical-align: middle;   
#  }
# </style>
# """))

# display: table-cell;
# text-align: center;
# vertical-align: middle;

# display: flex;
# align-items: center;
# text-align: left;

def plot_signals(samples):
    sns.set(font_scale=1)
    with plt.rc_context({'xtick.color':fg_color, 'ytick.color':fg_color}):
        # Temporary rc parameters in effect
        plt.figure(figsize=(20,9), facecolor='None')
        plt.subplots_adjust(bottom=0.1e-5)
        plt.suptitle("Amplitude vs Time: Audio Time Series for t=2 seconds",x=0.5,y=1.05,fontsize=22, color=fg_color)
        for i, f in enumerate(samples, 1):
            if i < 6:
                y, sr = librosa.load("two_second_audio/" + f, sr=SAMPLE_RATE)
                plt.subplot(2, 3, i)
                librosa.display.waveplot(y, sr=SAMPLE_RATE)
                plt.title(f.split("_")[0], fontsize=18, color=fg_color)
                plt.xlabel('Time (seconds)', fontsize=12, color=fg_color)
                plt.ylabel('Amplitude (dB)', fontsize=12, color=fg_color)
        plt.tight_layout()
                
            
def plot_class_distr(labels,labelShare):
    """" Pie Chart """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title('Count Distribution of Heart Conditions in Combined Dataset',y=1.08, fontsize=24)
    plt.pie(labelShare,labels=labels, autopct='%1.1f%%',startangle=120, textprops={'fontsize': 16})
    ax.set_xlabel('The majority of the audio are normal heart beats', fontsize=20, y=1.08)
    fig.set_facecolor('lightgrey')
    plt.show()
    
    
def plot_two_distr(class_labels, label_count, duration_labels, duration_count):
    txt="There is a disproportionate ratio \n of observations in each class"
    
    """" Pie Chart """
    fig = plt.figure(figsize=(22, 18))
    fig.text(0.37,0.020,txt, fontsize=25)
    sns.set(font_scale=1.7)
    colors = ['#55A868', '#C44E52', '#DD8452', '#4C72B0']
    total_time = sum(duration_count)
    total_count = sum(label_count)
    ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
    ax1.pie(label_count, labels=class_labels, autopct='%1.1f%%',startangle=120, radius = 1.2)
    ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)
    ax2.pie(duration_count, labels=duration_labels, 
            autopct=lambda p : '{:.1f}% ({:,.0f} sec)'.format(p,p * total_time/100), 
            startangle=320, radius = 1.2, colors=colors)
    ax1.set_title('Count Distribution of Heart Conditions\n in Combined Dataset',y=1.08, fontsize=30)
    ax2.set_title('Time Distribution of Audio\n in Combined Dataset',y=1.08, fontsize=30)
    
    fig.set_facecolor('lightgrey')
    plt.show()


def plot_time_dist(duration_labels, duration_count):
    """" Pie Chart """
    fig = plt.figure(figsize=(16, 13))   # smaller 12, 10
    sns.set(font_scale=1.2)
    colors = ['#55A868', '#C44E52', '#DD8452', '#4C72B0']
    total = sum(duration_count)
    ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
    ax1.pie(duration_count, labels=duration_labels, 
            autopct=lambda p : '{:.1f}%  ({:,.0f} sec)'.format(p,p * total/100), 
            startangle=140, radius = 1.2, colors=colors)
    ax1.set_title('Time Distribution of Audio\n in Expanded Training Dataset',y=1.08, fontsize=20)
    fig.set_facecolor('lightgrey')
    plt.show()
    
    
def plot_source_distr(l):
    """" Bar Plot """
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.set(style="whitegrid"); sns.set(font_scale=2)
    ax = sns.barplot(x="source", y="count", data=l, hue= "source", order=["a", "b"])
    ax.set_xlabel("source",fontsize=20)
    ax.set_ylabel("count",fontsize=20)
    ax.tick_params(labelsize=18)
    legend = ax.legend(title="source", loc = "upper left", prop={'size': 14})
    plt.title("Source of Combined Heartbeat Audio Dataset")
    #fig.set_facecolor('lightgrey')
    legend.texts[0].set_text("Clinical Trial in Hospitals")
    legend.texts[1].set_text("Crowdsourced via iPhone App")

    # annotate seaborn axis
    for p in ax.patches:
        if not np.isnan(p.get_height()):
            ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=18, color='black', xytext=(0, 20),
                     textcoords='offset points')
        _ = ax.set_ylim(0,550) # To make space for the annotations
        

def plot_keras_model(model, show_shapes=True,show_layer_names=True):
    return SVG(model_to_dot(model, show_shapes=show_shapes,
            show_layer_names=show_layer_names).create(prog='dot',format='svg'))
