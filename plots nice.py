import matplotlib.pyplot as plt
import numpy as np

# Function to create bar plots with different colors for each bar and a legend
def create_bar_plot(labels, values, title, legend_labels):
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size for better readability
    bar_width = 0.4  # Reduce the bar width
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))  # Use a colormap with distinct colors
    x = np.arange(len(labels))  # the label locations
    bars = ax.bar(x, values, width=bar_width, color=colors)
    
    # Set y-axis limits
    min_value = min(values) - 0.01
    max_value = max(values) + 0.01
    ax.set_ylim(min_value, max_value)
    
    # Add values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, round(yval, 3), ha='center', va='bottom', fontsize=10)
    
    # Avoid overlapping labels
    plt.xticks([])  # Remove x-ticks
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    plt.ylabel('Classification accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    plt.show()

# Experiment 1 data
normal_model = 0.933570581257414
semantic_hierarchical_model = 0.949584816132858
random_hierarchical_model = 0.919335705812574

labels_exp1 = ['Normal model', 'Semantic hierarchical model', 'Random hierarchical model']
values_exp1 = [normal_model, semantic_hierarchical_model, random_hierarchical_model]
legend_labels_exp1 = ['Normal', 'Semantic Hierarchical', 'Random Hierarchical']

create_bar_plot(labels_exp1, values_exp1, 'Experiment 1', legend_labels_exp1)

# Experiment 2 data
baseline_model_fasterrcnn = 0.933570581
baseline_model_retinanet = 0.928529063
baseline_model_yolov8 = 0.857058125
hierarchy_model_fasterrcnn = 0.949584816
hierarchy_model_retinanet = 0.957888493
hierarchy_model_yolov8 = 0.879650215

labels_exp2 = ['Baseline FasterRCNN', 'Hierarchy FasterRCNN', 'Baseline RetinaNet', 'Hierarchy RetinaNet', 'Baseline YOLOv8', 'Hierarchy YOLOv8']
values_exp2 = [baseline_model_fasterrcnn, hierarchy_model_fasterrcnn, baseline_model_retinanet, hierarchy_model_retinanet, baseline_model_yolov8, hierarchy_model_yolov8]
legend_labels_exp2 = ['Baseline FasterRCNN', 'Hierarchy FasterRCNN', 'Baseline RetinaNet', 'Hierarchy RetinaNet', 'Baseline YOLOv8', 'Hierarchy YOLOv8']

create_bar_plot(labels_exp2, values_exp2, 'Experiment 2', legend_labels_exp2)

# Experiment 3 data
semantic_hierarchical_model_2_layers = 0.949584816
semantic_hierarchical_model_3_layers = 0.959074733

labels_exp3 = ['Semantic hierarchical model 2 layers', 'Semantic hierarchical model 3 layers']
values_exp3 = [semantic_hierarchical_model_2_layers, semantic_hierarchical_model_3_layers]
legend_labels_exp3 = ['2 Layers', '3 Layers']

create_bar_plot(labels_exp3, values_exp3, 'Experiment 3', legend_labels_exp3)

# Experiment 4 data
root_nodes_post_grouped_5_super_classes = 0.959074733
root_nodes_grouped_5_super_classes = 0.96797153
root_nodes_post_grouped_3_super_classes = 0.984282325
root_nodes_grouped_3_super_classes = 0.987247924

labels_exp4 = ['Post grouped 5 super-classes', 'Grouped 5 super-classes', 'Post grouped 3 super-classes', 'Grouped 3 super-classes']
values_exp4 = [root_nodes_post_grouped_5_super_classes, root_nodes_grouped_5_super_classes, root_nodes_post_grouped_3_super_classes, root_nodes_grouped_3_super_classes]
legend_labels_exp4 = ['Post Grouped 5', 'Grouped 5', 'Post Grouped 3', 'Grouped 3']

create_bar_plot(labels_exp4, values_exp4, 'Experiment 4', legend_labels_exp4)

# Experiment 5 data
normal_5000 = 0.914972635
grouped_5000 = 0.916145426
normal_25000 = 0.928655199
grouped_25000 = 0.928264269

labels_exp5 = ['Normal 5000', 'Grouped 5000', 'Normal 25000', 'Grouped 25000']
values_exp5 = [normal_5000, grouped_5000, normal_25000, grouped_25000]
legend_labels_exp5 = ['Normal 5000', 'Grouped 5000', 'Normal 25000', 'Grouped 25000']

create_bar_plot(labels_exp5, values_exp5, 'Experiment 5', legend_labels_exp5)
