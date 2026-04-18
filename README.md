# Design, Optimization, and Comparative Evaluation of Modern YOLO Models for Real-World Object Detection

## Contents

1. **[Background and Motivation](#background-and-motivation)**
2. **[VisDrone Dataset](#visdrone-dataset)**
3. **[Baseline Model Training](#baseline-model-training)**
4. **[Baseline Model Results](#baseline-model-results)**
5. **[Controlled Experiment](#controlled-experiment)**
6. **[Model Improvement Cycles](#model-improvement-cycles)**
7. **[Multi-Version Comparison](#multi-version-comparison)**
8. **[Conclusion](#conclusion)**
9. **[Training Pipeline](#training-pipeline)**

## Background and Motivation

Real-time object detection in computer vision plays a central role in:

- Autonomous driving
- Robotics
- Smart surveillance
- Industrial inspection
- Smart cities

The You Only Look Once (YOLO) family, maintained by Ultralytics, is one of the most widely used real-time detection frameworks in industry. YOLOv11 and other modern versions aim to improve:

- Speed-accuracy trade-offs
- Small-object detection capabilities
- Model efficiency
- Training stability

This project focuses on building a complete experimental pipeline using [YOLOv11](https://docs.ultralytics.com/models/yolo11/), performing structured experiments, systematic analysis, and multi-version comparisons.

There are five parts to this project:

1. Training a Baseline Model
2. Loss Curve and Fitting Analysis
3. Experimental Design
4. Iterative Model Improvement
5. Multi-version YOLO Comparison 

## VisDrone Dataset

YOLOv11 nano (or, YOLOv11n) was trained as a baseline model using the VisDrone2019-DET trainset dataset. This dataset contains 6,471 aerial images taken from a drone. Each image has a corresponding annotation file (a `.txt` file), which contains the following columns: `bbox_left`, `bbox_top`, `bbox_width`, `bbox_height`, `score`, `category`, `truncation`, `occlusion`.

Prior to training, the VisDrone dataset needed to be converted to YOLO`s format, so several changes were made:

- Annotation columns were changed to YOLO's (`class_id`, `x_center`, `y_center`, `width`, and `height`)
- Bounding boxes were changed to be bound by their center points (`x_center`, `y_center`) + (`width`, `height`)
- Bounding boxes were normalized to the interval $[0, 1]$
- Class IDs were remapped from 1-11 to 0-10.
    - The categories are: `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`, `others`

Prior to remapping the class IDs, the dataset was filtered and cleaned:
- Any annotation whose class ID is 0 or greater than 11 is skipped.
- Any bounding box whose width or height is less than 0 is skipped.

All extra annotation fields (`score`, `truncation`, and `occlusion`) are deleted.

#### Example: Annotation Conversion Before & After

**VisDrone Annotation**
```txt
891,578,74,35,1,4,0,0
996,569,75,35,1,4,0,0
1143,605,75,41,1,4,0,0
705,693,76,44,1,4,0,0
1274,628,80,37,1,4,0,0
1681,624,88,45,1,4,0,0
1588,736,81,65,1,4,0,0
1147,811,88,56,1,4,0,0
1897,537,78,37,1,4,0,0
1694,738,85,49,1,5,0,0
1498,759,78,46,1,5,0,0
```

**YOLO Annotation**
```txt
3 0.463500 0.397333 0.037000 0.023333
3 0.516000 0.390000 0.037500 0.023333
3 0.608250 0.416667 0.037500 0.027333
3 0.371000 0.476000 0.038000 0.029333
3 0.657000 0.430000 0.040000 0.024667
3 0.862500 0.431000 0.044000 0.030000
3 0.814250 0.512000 0.040500 0.043333
3 0.595500 0.560333 0.044000 0.037333
3 0.968500 0.369667 0.039000 0.024667
4 0.869250 0.508667 0.042500 0.032667
4 0.768500 0.521333 0.039000 0.030667
```

## Baseline Model Training 

For the baseline model, YOLOv11n was used because it is trained quickly (< 24 minutes) and sets a lower-bound point of reference to use later when comparing against an improved version and other versions of YOLO.

After the annotation files have been converted, an 80/20 split is done on the images for training and validation.
- 5,177 images are used for training
- 1,294 are used for validation

The model was optimized to train on an Nvidia A100 GPU via Google Colab. The following hyperparameters were used:


| Hyperparameter                  | Value                             | Notes                                                                                                                                                                                                                                                                                                                             |
| ------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Epochs                          | 50                                | Should be enough to observe convergence trends |
| Image Size                      | 640                               | YOLO's standard res                                                                                                                                                                                                                                                                                                               |
| Batch Size                      | 64/128                            | Dependent on the GPU's RAM. For models with 40GB, images are processed in batches of 64 at a time. For models with 80GB, 128 are processed at a time.                                                                                                                                                                             |
| Workers                         | 8                                 | The number of worker threads on the CPU that pre-load and pre-process the next batch of images while the GPU is training off of the current batch.                                                                                                                                                                                |
| Cache                           | RAM                               | All images are initially cached into RAM so that they don't have to be reloaded at the start of each epoch.                                                                                                                                                                                                                       |
| Automatic Mixed Precision (AMP) | `True`                            | Most values are used and stored as 16-bit floats instead of 32-bit because A100 tensor cores are optimized for 16-bit (this gives roughly 2x the throughput). Some critical values (e.g., loss scaling) are automatically kept as 32-bit floats to prevent them from underflowing or overflowing.                                 |
<!-- 
| Learning Rate Schedule          | cosine                               | This controls how the learning rate changes over time. During early training stages, a larger learning rate is ideal to escape the random initial weights. Later in the training stage, a smaller learning rate is ideal to prevent overshooting. The learning rate is smoothly reduced by following the shape of a cosine curve. |
-->

- Epochs: 50
- Image Size: 640 (this is YOLO's standard res)
- Batch Size: 64/128

## Baseline Model Results

Using an Nvidia A100 GPU with 40GB of RAM, the model took ~24 minutes to train. All result data was stored in [/results/yolov11n_baseline/](/results/yolov11n_baseline/). 

This includes:
- [args.yaml](/results/yolov11n_baseline/args.yaml): Contains all of the Ultralytics hyperparameters and settings used during training.
- [baseline_summary.csv](/results/yolov11n_baseline/baseline_summary.csv): Contains a summary of the baseline model's training results (info on the dataset, number of epochs, final training loss, etc.)
- [results.csv](/results/yolov11n_baseline/results.csv): Contains the data on training loss, validation loss, mAP@0.50, mAP@0.50:0.95, precision, and recall for each epoch. This is the data that was used to generate all plots and graphs.
- Plots that are auto-generated by Ultralytics, such as a [confusion matrix](/results/yolov11n_baseline/confusion_matrix.png).
- Plots that I generated for analysis, such as a [composite loss curve](/results/yolov11n_baseline/composite_loss_curve.png).


### Training Loss & Validation Loss

YOLOv11 splits loss into three components:

1. Bounding Box Regression Loss (Box Loss): Measures how accurately the model's predicted bounding box coordinates match the ground truth bounding box coordinates.

2. Classification Loss: Measures how well the model was able to assign classification labels to each detected object.

3. Distribution Focal Loss (DFL): Since object edges can be ambiguous due to blurry boundaries or occlusion from another object, the YOLOv11 doesn't predict a single exact box edge coordinate. Instead, it predicts a distribution over possible locations where the edge of the bounding box could be.

Training loss is computed on the training sets during a forward pass. Validation loss is computed after the last 20% of images are used during the validation stage, after an epoch has completed.

<img width="2067" height="603" alt="image" src="https://github.com/user-attachments/assets/f1269ac6-26a2-42ae-8a67-de4bddfb8db2" />

#### Box Loss

Training loss began a bit over 2.0 and decreased steeply for the first ~6 epochs, then continued to fall steadily even in the later epochs (though it clearly slowed down to some degree). Epoch 50 ended with a loss of 1.4. It may be worth testing with 60 or 65 epochs to see how close it is to plateauing.

Validation loss began at just over 1.9 and followed a similar start trajectory to training loss, where it experienced a sharp drop for the first ~6 epochs, then continued to fall steadily. However, around epoch 25, it plateaued and didn't move much for the last half of the epochs. Epoch 50 ended with a loss of a bit under 1.5.

#### Classification Loss

Training loss began just over 3.5 and dropped sharply to ~1.25 by epoch 10, then severely plateaued for the rest of the epochs. Epoch 50 ended with a loss just under 1.0.

Validation loss followed a nearly identical trajectory to the training loss. That said, it started much lower than the training loss; epoch 1's loss was ~1.6. Epoch 50 ended with a loss just above 1.0.

#### DFL

Training loss began just around 1.5 and severely dropped for the first ~15 epochs, then plateaued once it reached a loss of ~0.91 around epoch 26.

Validation loss follows a very similar path and stays just above the training loss curve for nearly every epoch. It begins with a low value just over 0.98, sees a short spike during epochs 3-5, then has a gradual decline for nearly the rest of the training period.

### Mean Average Precision (mAP), Precision, and Recall

In order to determine if convergent behavior was displayed, the standard deviation of the validation loss over the last 10 epochs was calculated against a threshold of $0.005$. For baseline training, the standard deviation of the validation loss over the last 10 epochs was $0.01517$, indicating the model had not yet fully converged by epoch 50. That said, the declining trend in the final epochs suggests convergence was approaching.

<img width="1287" height="637" alt="image" src="https://github.com/user-attachments/assets/52edad7a-1068-4086-94ed-d523ef1dd20f" />

The composite loss curve shows a clear plateau around epoch 18. It may be worth training the model for an additional 10 epochs to improve the box loss, but with the classification loss and DFL loss both having plateaued already, this could easily cause convergence to occur.

<img width="1807" height="643" alt="image" src="https://github.com/user-attachments/assets/1e96a3fe-edd1-4f66-8dc8-08008db1595c" />

#### Difference Between Validation Loss & Training Loss Per Epoch

The blue bars indicate epochs where the validation loss is lower than the training loss, and the orange bars indicate epochs where the opposite is true. Starting at epoch 40 and onward, there are only orange bars, and they are consistently growing larger, which may indicate that overfitting is beginning to occur.

#### Smoothed Loss Curves

This graph shows the composite training loss and validation loss curve from the graph above as a smoothed curve with a window size of 5.

---

The mAP metric evaluates how well the model is able to find objects and how correctly it is at identifying the class that an object belongs to. Precision measures how correct the model was in its predictions. Recall measures how many objects the model was able to find out of all of the objects that are in the image.

There are two mAP metrics that are evaluated:

- mAP@0.50: Measures the mAP at a fixed IoU threshold of 0.50. This tells us if the model is able to identify objects at all.
- mAP@0.50:0.95: Measures the average mAP across 10 different IoU thresholds ranging from 0.50 to 0.95 (with a step of 0.05). This metric is stricter and evaluates how precise the model is at object detection.

<img width="1806" height="643" alt="image" src="https://github.com/user-attachments/assets/8640fd4d-5a21-41ab-a82d-734328c8d11d" />

#### Mean Average Precision (mAP)

Both curves in the mAP graph rise steeply for the first ~12 epochs, which is normal as the model quickly learns from the pretrained weights, adapting to VisDrone. Growth slows after epoch ~24, and both curves begin to plateau around epoch 45. That said, neither curve has fully flattened by epoch 50, which is consistent with the "not yet converged" diagnosis. At the end of the training period, there is a large gap between mAP@0.50 and mAP@0.50:0.95. This indicates that the model is good at finding objects, but it can't draw very precise boxes around them.

#### Precision & Recall

The precision and recall lines follow each other and consistently increase during the training period. Towards the end of the training period, their growth is slow, but it hasn't plateaued yet. Precision ends around 41%, and recall ends around 33%, meaning that the model is more conservative with its box predictions. When the model draws a box around an object, prediction is usually correct, but it also misses a few boxes in each image as a result.

### Overall Baseline Results

<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/01046680-8f44-417a-868c-fd30ed1f0cf6" />

Overall, class loss converged the fastest, while box loss still has room for improvement. There is some evidence of overfitting, but nothing too serious. The YOLOv11n model is the smallest YOLOv11 variant. This is important because the dataset contains several small objects (e.g., pedestrians), which are hard for the model to pick up on. 

## Controlled Experiment

### Experimental Settings
For my controlled experiment, I decided to explore how doubling the image resolution from 640 px to 1280 px would improve the model's detection precision. To handle larger resolution sizes, the batch size was reduced by 75% to 16/32 (depending on the amount of RAM the GPU has). For my experiment, I used an A100 GPU with 80GB of RAM, so my batch size was 32. All other hyperparameters and other settings remained the same as they did during the baseline training.

### Results

All of the experimental model's individual results can be found in [results/yolov11n_1280px](/results/yolov11n_1280px/).

| Metric                       | Baseline Model | Experimental Model |
| ---------------------------- | -------------- | ------------------ |
| mAP@0.50                     | 0.299          | 0.472              |
| mAP@0.50:0.95                | 0.169          | 0.287              |
| Precision                    | 0.413          | 0.576              |
| Recall                       | 0.329          | 0.467              |
| F1 Score                     | 0.35           | 0.50               |
| Params (M)                   | 2.58           | 2.58               |
| FLOPs (G)                    | 8.1            | 6.3                |
| Model Size (MB)              | 5.5            | 5.6                |
| Training Time (min)          | ~24            | ~66                |
| GPU & GPU RAM                | A100, 42GB     | A100, 80GB         |
| Inference Speed <br>(ms/img) | 0.3            | 1.0                |

<img width="1807" height="643" alt="image" src="https://github.com/user-attachments/assets/efa90336-5b33-40e0-ae25-31052f6caf26" />

The experimental model's training loss and validation loss follow a very similar curve to the baseline's. The model's best epoch was much closer to the end of the training period, so there may be more room for improvement with a larger number of epochs.

<img width="2327" height="643" alt="image" src="https://github.com/user-attachments/assets/49632008-05cc-4b7c-ad90-045ac119c326" />


Though both models' loss curves are of a similar shape, there is a clear improvement with the experimental model's results. Its validation loss, mAP@0.50, and map@0.50:0.95 are all significantly better than the baseline's.

There are other important considerations to make, other than the experimental model's object detection abilities. A GPU with nearly double the amount of RAM as the one that trained the baseline model was used to train the experimental model, but the training time was 2.75x slower than the baseline's. Furthermore, the experimental model's inference speed is ~3.33x slower than the baseline. These metrics really matter when it comes down to your use case of the model and whether you care more about accuracy or speed.

## Model Improvement Cycles

I conducted two rounds of improvement cycles. because I thought the model still has room to improve. For the first round, I wanted to try improving the model by adding an additional 15 epochs (65 total). Though there were signs of convergence beginning and potential overfitting, I felt that there was still room for the model to grow.

For the second round, I wanted to see if a different optimization algorithm could reduce how much the model plateaued. An optimization algorithm is used to update the model's weights during the training period. Since I used `optimizer='auto'` when I trained the model, Ultralytics selected the AdamW optimizer. This optimization algorithm determines an individual parameter's step size. It tracks a smoothed average of recent gradients and a smoothed average of squared gradients (these act as a measure of how volatile a parameter's gradient is). The algorithm causes noisy gradients to become smaller and smaller gradients to become larger. The algorithm also slowly shrinks all of the parameters towards zero during training so that the model doesn't overfit to individual weights. This leads to a quick drop 

Stochastic Gradient Descent (SGD) is another optimization algorithm. SGD works by updating weights using gradients from small batches of training samples, with momentum helping to stabilize and accelerate convergence. I would like to see how the model performs with SGD because the algorithm has historically been favored for object detection. SGD often produces more stable and gradual progress, compared to AdamW, which usually produces faster progress more aggressively, so I think it is fair to give the model a bit more time to train.

### Experimental Model Results vs. Improvement Cycle 1 Results 

| Metric                       | Experiment Model   | Improvement Cycle 1 |
| ---------------------------- | ------------------ | ------------------- |
| mAP@0.50                     | 0.472              | 0.481               |
| mAP@0.50:0.95                | 0.287              | 0.291               |
| Precision                    | 0.576              | 0.582               |
| Recall                       | 0.467              | 0.475               |
| F1 Score                     | 0.50               | 0.51                |
| Params (M)                   | 2.58               | 2.58                |
| FLOPs (G)                    | 6.3                | 6.3                 |
| Model Size (MB)              | 5.6                | 5.6                 |
| Training Time (min)          | ~66                | ~85                 |
| GPU, VRAM                    | A100, 80GB         | A100-SXM4, 85.1GB   |
| Inference Speed <br>(ms/img) | 1.0                | 1.0                 |

#### Improvement Model's Results

<img width="1905" height="1204" alt="image" src="https://github.com/user-attachments/assets/7cfcb146-f325-4802-82bd-084c4c4dbfc4" />

While the improvement model performs slightly better than the experiment model, these results suggest that there is still some room for improvement. None of the curves shows significant plateauing, and there are no large gaps between the training and validation curves. Precision is still fairly higher than recall, meaning the model is still able to accurately detect objects, but the detected boxes aren't always localized with great precision. Increasing the number of epochs to 120-130 with a patience of 20-25 may be needed to see how far the model's training can be stretched out before it plateaus.

#### Experiment Model vs. Improvement Cycle 1

<img width="1807" height="643" alt="image" src="https://github.com/user-attachments/assets/bc7a2abf-8a47-42e0-938e-49ba7e6e6d8b" />

There is little improvement in the model's loss overall. The model still began to plateau towards the end of the training period.

---

### Improvement Cycle 1 Results vs. Cycle 2 Results

| Metric                       | Improvement Cycle 1 | Improvement Cycle 2 |
| ---------------------------- | ------------------- | ------------------- |
| mAP@0.50                     | 0.481               | 0.486               |
| mAP@0.50:0.95                | 0.291               | 0.296               |
| Precision                    | 0.582               | 0.593               |
| Recall                       | 0.475               | 0.475               |
| F1 Score                     | 0.51                | 0.52                |
| Params (M)                   | 2.58                | 2.58                |
| FLOPs (G)                    | 6.3                 | 6.3                 |
| Model Size (MB)              | 5.6                 | 5.6                 |
| Training Time (min)          | ~85                 | ~80                 |
| GPU, VRAM                    | A100-SXM4, 85.1GB   | A100, 80GB          |
| Inference Speed <br>(ms/img) | 1.0                 | 1.0                 |


#### Improvement Model's Results

<img width="1905" height="1204" alt="image" src="https://github.com/user-attachments/assets/2d7eab7f-5124-484f-bd3b-529a7c49ac47" />

These results are very similar to the previous cycle's results versus the experiment model's. They suggest that there is still room for improvement; none of the curves have begun to plateau, and there are no large gaps between the training and validation curves. There is still a gap between precision and recall.

#### Improvement Cycle 1 vs. Improvement Cycle 2

<img width="1807" height="643" alt="image" src="https://github.com/user-attachments/assets/c8343b14-5058-485a-8019-8a91337e0103" />


Both models converge to roughly the same loss, but the cycle 2 model increases slightly during the last few epochs. This may be a small noise spike, or it could indicate that there is still room for improvement before plateauing. SGD generalizes slightly better than AdamW, which often causes the loss to drop more slowly and gradually, but leaves room for finding a flatter minimum.

<img width="2327" height="643" alt="image" src="https://github.com/user-attachments/assets/7b751914-b74b-4831-bd74-3870f3dae046" />

Both curves on all three graphs begin to plateau around epoch 56. I think this cycle strengthens the evidence that training with ~120 epochs and a patience of ~20 may be a good next step to take.

## Multi-Version Comparison

### Models

I compared the baseline YOLOv11n model against three other YOLO versions:

- YOLOv8 nano (YOLOv8n)
- YOLOv9 tiny (YOLOv9t; the smallest variant of the v9 family, just like nano is the smallest of the v11 family)
- YOLOv10 nano (YOLOv10n)

### Comparison

| Metric                       | YOLOv8n     | YOLOv9t     | YOLOv10n    | YOLOv11n   |
| ---------------------------- | ----------- | ----------- | ----------- | ---------- |
| mAP@0.50                     | 0.293       | 0.291       | 0.288       | 0.299      |
| mAP@0.50:0.95                | 0.166       | 0.165       | 0.162       | 0.169      |
| Precision                    | 0.394       | 0.417       | 0.402       | 0.413      |
| Recall                       | 0.327       | 0.318       | 0.317       | 0.329      |
| F1 Score                     | 0.34        | 0.34        | 0.34        | 0.35       |
| Params (M)                   | 3.01        | 1.97        | 2.27        | 2.58       |
| FLOPs (G)                    | 8.1         | 7.6         | 6.5         | 6.5        |
| Model Size (MB)              | 6.2         | 4.6         | 5.7         | 5.5        |
| Training Time (min)          | ~59         | ~65         | ~88         | ~24        |
| GPU & GPU RAM                | L4, 23.7 GB | L4, 23.7 GB | L4, 23.7 GB | A100, 42GB |
| Inference Speed <br>(ms/img) | 0.8         | 1.0         | 1.1         | 0.3        |

*Note*: When I trained the three comparison models, there were no A100 GPUs available, so Colab automatically connected me to a runtime with an L4 GPU instead. Thus, the training time results are not very useful, but I have included them anyway to see how long the three comparison models took against each other.

---

<img width="2327" height="669" alt="image" src="https://github.com/user-attachments/assets/89438bd4-a65e-4190-ad84-2880501b8248" />

| Metric                       | YOLOv8n     | YOLOv9t     | YOLOv10n    | YOLOv11n   |
| ---------------------------- | ----------- | ----------- | ----------- | ---------- |
| mAP@0.50                     | 0.293       | 0.291       | 0.288       | **0.299**  |
| mAP@0.50:0.95                | 0.166       | 0.165       | 0.162       | **0.169**  |
| Precision                    | 0.394       | **0.417**   | 0.402       | 0.413      |
| Recall                       | 0.327       | 0.318       | 0.317       | **0.329**  |

The baseline v11n model performed the best in three of the four metrics, but the spread for each metric is extremely minimal; no model is meaningfully better than another.

--- 

<img width="1807" height="802" alt="image" src="https://github.com/user-attachments/assets/24d2a0ba-17c7-4539-80cb-863f7066db75" />

#### Model Size vs. Accuracy

| Metric          | YOLOv8n | YOLOv9t | YOLOv10n | YOLOv11n |
| --------------- | ------- | ------- | -------- | -------- |
| mAP@0.50        | 0.293   | 0.291   | 0.288    | **0.299**|
| Model Size (MB) | 6.2     | **4.6** | 5.7      | 5.5      |

YOLOv11n is the most accurate and the second smallest. That said, there is still not a very big difference between the most accurate and least accurate models, so any model could realistically be chosen against these metrics.

#### Total Speed

| Metric               | YOLOv8n | YOLOv9t | YOLOv10n | YOLOv11n |
| -------------------- | ------- | ------- | -------- | -------- |
| Total Time (ms/img)  | 6.47    | 7.77    | **4.57** | 7.32     |
| Throughput (img/sec) | ~155    | ~129    | **~219** | ~137     |

YOLOv10n's post-processing time is significantly shorter than the other models'. This is because YOLOv10 removed non-maximum suppression (NMS). NMS is a post-processing technique that removes redundant boxes that overlap each other for the same object.

## Conclusion

The improvement model performed the best out of all the models I trained. There are signs that it still has room for improvement, but it may be worth testing the experimental model with more epochs first, to see if there is any improvement left for it. The largest increase in performance was between the baseline and experiment models, with the image's resolution being the most significant contributor. This is because of the high number of small objects in the dataset, which can be hard to detect with low image resolution. The largest tradeoff with increasing resolution was the amount of time to train the model. None of the models suffered from overfitting or underfitting, but several did show signs of plateauing. Further improvements could be made, such as increasing the image resolution even more, using a bigger version of the model (e.g., small, medium, etc.), and/or increasing the number of epochs.

## Training Pipeline

There are two Jupyter Notebooks needed to reproduce this study:

- [yolov11n.ipynb](/yolov11n.ipynb)
- [yolo_multi_version_comparison](/yolo_multi_version_comparison.ipynb)

The notebooks are configured to run in Google Colab, so they will need to be modified if you want to run them locally.

### Steps to Reproduce

1. Download the [VisDrone-DET2019 trainset dataset](https://github.com/VisDrone/VisDrone-Dataset) (1.44 GB) and place the .zip file in `My Drive` in Google Drive.
    - The path to this file should be `/content/drive/MyDrive/VisDrone2019-DET-train.zip`.
2. In Google Drive, create a folder called `YOLOv11-Project`
    - You use a different name, but make sure to update the notebooks so that they use the correct folder.
3. Clone this repository, or download the notebooks individually, then open them in Google Colab.
4. Run the `yolov11n.ipynb` notebook first. This notebook does the following:
    1. Unzips the dataset locally in the VM and cleans the data into a YOLO-compatible format
    2. Trains the baseline YOLO model, analyzes the results, and makes graphs and tables using the results. The model's results and graphs are saved in Google Drive at `/YOLOv11-Project/runs/yolov11n_baseline/`
    3. Trains the model for the controlled experiment, analyzes the results, and makes graphs and tables using the results. Graphs are also generated to compare the experiment model's results to the baseline's. The model's results and graphs are saved in Google Drive at `/YOLOv11-Project/runs/yolov11n_1280px/`
    4. Trains the improvement model, analyzes the results, and makes graphs and tables using the results. Graphs are also generated to compare the improvement model's results to the experiments' results. The model's results and graphs are saved in Google Drive at `/YOLOv11-Project/runs/yolov11n_improvement/`
5. Run the `yolo_multi_version_comparison.ipynb` notebook. This notebook does the following:
    1. Unzips the dataset locally in the VM and cleans the data into a YOLO-compatible format
    2. Trains YOLOv8n, YOLOv9t, and YOLOv10n on the same dataset as YOLOv11n with the same hyperparameters. The results of each model are saved at `/YOLOv11-Project/runs/model_name_comparison/` (e.g., `/YOLOv11-Project/runs/yolov8n_comparison/`).
    3. Results of all four models are loaded into the runtime, and analysis is conducted to produce a comparison graph and other information. The comparison graphs are saved at `/YOLOv11-Project/multi-version-comparison/`.

*Note*: All of my results can be found in this repository in [results.zip](/results.zip).
