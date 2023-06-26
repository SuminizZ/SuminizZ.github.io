---
layout: post
title : "[Paper Review & Implementation] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Faster R-CNN, 2015)"
img: papers/objd/faster_rcnn.png
categories: [papers-objd]  
tag : [Paper Review, Object Detection, Faster R-CNN, PyTorch]
toc : true
toc_sticky : true
---

## **Outlines**
- [**Reference**](#reference)
- [**Faster R-CNN vs Fast R-CNN vs R-CNN**](#faster-r-cnn-vs-fast-r-cnn-vs-r-cnn)
- [**Step By Step Implementation of Faster R-CNN with PyTorch**](#step-by-step-implementation-of-faster-r-cnn-with-pytorch)
- [**Attention of Transformer**](#attention-of-transformer)
- [**Embedding and Positional Encoding**](#embedding-and-positional-encoding)
- [**Encoder and Decoder Architecture**](#encoder-and-decoder-architecture)
- [**Comparisoin of Computational Efficiency to Other Models**](#comparisoin-of-computational-efficiency-to-other-models)
- [**Performance of Transformer in Machine Translation**](#performance-of-transformer-in-machine-translation)

<br/>

## **Reference**

<br/>

- [**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, Shaoqing Ren, 2015**](https://arxiv.org/abs/1506.01497){:target="_blank"}
- [**How FasterRCNN works and step-by-step PyTorch implementation**](https://www.youtube.com/watch?v=4yOcsWg-7g8){:target="_blank"}

<br/>

## **Faster R-CNN** vs **Fast R-CNN** vs **R-CNN**

<br/>

&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/91341de2-0c34-40be-a6b3-d6a8b42f4ec7" width="700">

<br/>

- Faster R-CNN is an improvement of R-CNN and Fast R-CNN, integrating region proposal network (RPN) into the network architecture unlike R-CNN and fast R-CNN that adopt external RPN with selective search algorithm.

<br/>

- **R-CNN**
    
    - As an initial model of R-CNN series, R-CNN used a selective search algorithm to generate region proposals outside of the networks and then feed each of these proposals into a distinct CNN networks for feature extraction. 

    - These features are then subsequently passed to a set of FC layers to perform classification task and bounding box regression task, separately. 

    - Due to this multi-stage pipeline, R-CNN suffered from mutiple sets of computationally expensive operations included in the model architecture. 

- **Fast R-CNN** 

    - Also adopts external region proposals using selective search. 

    - Unlike R-CNN, Fast R-CNN shares one CNN across the entire image and then applies an RoI (Region of Interest) pooling layer to extract features all at once from a single output of the CNN. 

    - Integrating multiple CNNs into a single shared CNN significantly saves the amount of computations required. 

    - In addition, Fast R-CNN jointly optimizes the classification and regression tasks using multi-task loss, which is the sum of classification and regression loss. 

- **Faster R-CNN**

    &emsp;&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/98fec387-b682-4759-b5c6-5661fbd50bd4" width="600">

    - Faster R-CNN uses internal RPN, which shares CNN with object detection networks. This RPN directly generates region proposals from extracted feature maps and passes them into NMS and proposal target layers to get final region proposals.

    - Multi-task loss is computed between the locations and labels of bounding boxes predicted from RPN and ground truth bounding boxes. 

    - NMS layer and proposal target layer use initially defined anchor boxes and computed ojectness scores from RPN.
    
    - Then, applies RoI pooling to output feature map and extract final roi features based on the target proposals gained from proposal target layers.
    

<br/>


## **Step By Step Implementation of Faster R-CNN with PyTorch**

<br/>

- [**Full codes here**](https://github.com/SuminizZ/Implementation/tree/main/Faster_R-CNN)

<br/>

### **1. Feature Extractor VGG-16**

<br/>

- Implement pre-trained VGG-16 networks as a feature extractor to get output feature maps with desired sub sampling size, which in this case, 50.

- Firstly starts with 3 x 800 x 800 image and downsample it into 512 feature maps with 50 x 50 size. (final output shape : 512 x 50 x 50)

<br/>

```python
class FeatureExtractor(nn.Module):
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()

        model = torchvision.models.vgg16(pretrained=True).to(device)
        features = list(model.features)

        dummy = torch.zeros((1, 3, 800, 800)).float()    # test image array
        req_features = []
        dummy = dummy.to(device)

        for feature in features:
            dummy = feature(dummy)
            if dummy.size(2) < 800//16:     # 800/16=50
                break
            req_features.append(feature)
            out_channels = dummy.size(1)

        self.feature_extractor = nn.Sequential(*req_features)

    def forward(self, x):
        return self.feature_extractor(x)
```

<br/>

- Input 

    - x : 3 x 800 x 800 images


<br/>

### **2. Anchor Generation Layers**

<br/>

```python
def generate_anchors(feature_size):
    ctr = torch.empty((feature_size**2, 2))
    ctr_x = torch.arange(16, (feature_size + 1) * 16, 16)
    ctr_y = torch.arange(16, (feature_size + 1) * 16, 16)

    anc = 0
    for x in ctr_x:
        for y in ctr_y:
            ctr[anc, 0] = x - 8
            ctr[anc, 1] = y - 8
            anc += 1

    ratios = [0.5, 1, 2]
    scales = [8, 16, 32]
    combs = [(r, s) for r in ratios for s in scales]
    base_size = 8

    i = 0
    anchor_boxes = torch.empty((feature_size**2*9, 4))
    for x, y in ctr:
        for z in combs:
            ratio, scale = z
            w = scale*base_size*np.sqrt(ratio)
            h = scale*base_size*(1/np.sqrt(ratio))
            anchor_boxes[i] = torch.Tensor([x-w, y-h, x+w, y+h])
            i += 1

    return anchor_boxes
```

<br/>

```python
def select_valid_anchors(anchor_boxes):
    valid_idxs = np.where((anchor_boxes[:,0] >= 0) &
                          (anchor_boxes[:,1] >= 0) &
                          (anchor_boxes[:,2] <= 800) &
                          (anchor_boxes[:,3] <= 800))[0]
    valid_anchors = anchor_boxes[valid_idxs]
    return valid_idxs, valid_anchors
```

<br/>

- Argument
    
    - feature size : 50

- Generates candidate anchor bounding boxes with various sizes and ratios. 

    - sub-sampling rate = 1/16

    - image size : 800 x 800

    - sub-sampled feature map size : 800 x 1/16 = 50

    - 50 x 50 = 2500 anchors and each anchor generate 9 anchor boxes with varying scales (3) and ratios (3)

    - total anchor boxes = 50 x 50 x 9 = 22500

- Select only valid anchors whose 4 coordinates are within 800 x 800 initial image size. 

<br/>

### **3. Compute Intersection Over Union (IoU) with Ground-Truth Boxes**

<br/>

```python
def get_IoUs(bbox, valid_anchors):
    ious = torch.ones((len(valid_anchors), bbox.shape[0]))*-1

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        ax1, ay1, ax2, ay2 = valid_anchors[:, 0], valid_anchors[:, 1], valid_anchors[:, 2], valid_anchors[:, 3]
        max_x1 = np.maximum(ax1, x1)
        max_y1 = np.maximum(ay1, y1)
        min_x2 = np.minimum(ax2, x2)
        min_y2 = np.minimum(ay2, y2)

        idxs = np.where((max_x1 < min_x2) & (max_y1 < min_y2))[0]
        tot_area = (ax2[idxs] - ax1[idxs])*(ay2[idxs] - ay1[idxs]) + (x2-x1)*(y2-y1)
        overlapped_area = (min_x2[idxs] - max_x1[idxs])*(min_y2[idxs] - max_y1[idxs])
        ious[idxs, i] =  overlapped_area / (tot_area - overlapped_area)

    # print(ious[8930:8940, :])
    return ious
```

<br/>

- Function that computes IoU with ground truth boxes. 

- IoU = overlapped area / total area

    - total area = anchor box area + ground truth box area - overlapped area

- Used to set labels for each anchor box and later be used in NMS and proposal target layers.


<br/>

### **4. Get Lables and Parameterized Coordinates of Anchors**

<br/>

```python
def get_anchor_labels(ious, valid_anchors):
    labels = torch.ones(len(ious))*-1

    max_iou_idxs = ious.argmax(axis=0)    # highest IoU with each ground truth box 
    max_iou = ious[max_iou_idxs, torch.arange(ious.shape[1])] - 0.01  
    print(max_iou)
    labels[np.where(ious >= max_iou)[0]] = 1
    print(np.where(ious >= max_iou)[0])

    max_iou_gtb_idxs = ious.argmax(axis=1)
    max_iou_gtb_vals = ious[torch.arange(len(max_iou_gtb_idxs)), max_iou_gtb_idxs]

    pos_label_threshold = 0.7
    neg_label_threshold = 0.3

    labels[max_iou_gtb_vals >=  pos_label_threshold] = 1     # have Io
    labels[max_iou_gtb_vals <  neg_label_threshold] = 0

    max_iou_bbox = bbox[max_iou_gtb_idxs]

    n_sample = 256
    pos_ratio = 0.5
    n_pos = pos_ratio * n_sample

    pos_index = np.where(labels == 1)[0]
    print(len(pos_index))
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index,
                                        size = (len(pos_index) - n_pos),
                                        replace=False)
        labels[disable_index] = -1

    return labels
```

<br/>

- Get labels for every valid anchor boxes.

    - Set label = 1 if

        1. anchors with highest IoU with each ground truth box
        
        2. IoU with respect to gt boxes higher than 0.7

    - label = 0 if IoU less than 0.3

    - otherwise, all are set to be -1, which will be ignored

    - Fix the upper limit of the number of positive labels as 128. If the number of positively labeled boxes is bigger than the limit, randomly de-select them untill it reaches to the limit. 

- Get parameterized locations of anchor boxes.

    - Parameterizes the 4 coordinates of anchor boxes with each one's closest ground-truth boxes.