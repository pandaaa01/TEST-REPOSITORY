from pre_defined_functions import *
import pandas as pd


def task_caller(data, identified_task, target_data):
    if identified_task == "line plot":
        return lineplot(data, target_data[0], target_data[1])
    if identified_task == "pie chart":
        return pie_chart(data, target_data[0])
    if identified_task == "median":
        return median(data, target_data[0])
    if identified_task == "mean":
        return mean(data, target_data[0])
    if identified_task == "bar graph":
        return bar_graph(data, target_data[0])
    if identified_task == "correlation":
        return correlation(data, target_data[0], target_data[1])
    if identified_task == "standard deviation":
        return standard_deviation(data, target_data[0])
    if identified_task == "histogram":
        return histogram_plot(data, target_data[0])

