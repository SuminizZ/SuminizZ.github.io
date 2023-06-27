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
    - [**0. Sample Image Data and Ground-Truth Boxes**](#0-sample-image-data-and-ground-truth-boxes)
    - [**1. Feature Extractor VGG-16**](#1-feature-extractor-vgg-16)
    - [**3. Compute Intersection Over Union (IoU) with Ground-Truth Boxes**](#3-compute-intersection-over-union-iou-with-ground-truth-boxes)
    - [**4. Get Lables and Parameterized Coordinates of Anchors**](#4-get-lables-and-parameterized-coordinates-of-anchors)
    - [**5. Region Proposal Networks (RPN)**](#5-region-proposal-networks-rpn)
    - [**6. Multi-Task Loss**](#6-multi-tsk-loss)
    - [**7. Proposal Layer : NMS & Proposal Target Layers**](#7-proposal-layer--nms--proposal-target-layers)
    - [**8. Get Ground-Truth Labels and Paramterized Locations of Selected Anchors**](#8-get-ground-truth-labels-and-paramterized-locations-of-selected-anchors)
    - [**9. RoI Pooling**](#9-roi-pooling)
    - [**10. Fast R-CNN**](#10-fast-r-cnn)

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

### **0. Sample Image Data and Ground-Truth Boxes**

<br/>

- An image data with shape (3 x 800 x 800) that will used and its 4 ground truth bounding boxes that localizes the target objects. 

&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/5998e933-6c04-4e6c-a8b8-ec28c1d02708" width="470">

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

&emsp;&emsp;**Display some of the generated anchor boxes (<span style='color:red;'>red</span>) and the ground truth boxes (<span style='color:green;'>green</span>)**

&emsp;&emsp;&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/4e04e24e-7b6f-4a8b-9458-b52274f4fb5b" width="500">


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

- Used to assign labels for each anchor box and later be used in NMS and proposal target layers.


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

    max_iou_gtb_idxs = ious.argmax(axis=1)    # index of ground truth box of highest IoU with each anchor box
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

- Get **labels** for every valid anchor boxes.

    - Set label = 1 (object) if

        1. Anchor boxes that have highest IoU with each ground truth box.
        
        2. Each anchor's highest IoU >= 0.7

    - label = 0 (background) if IoU less than 0.3 

    - otherwise, all are set to be -1, which will be ignored in further steps.

    - Fix the upper limit of the number of positive labels as 128. If the number of positively labeled boxes is bigger than the limit, randomly de-select them untill it reaches to the limit. 

- Get **parameterized locations** of anchor boxes.

    - Parameterizes the 4 coordinates of anchor boxes with each one's closest ground-truth boxes.

        <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/441f95de-d867-400e-a40c-084062597999" width="400">

    - $\large x,\, y,\, w,\, h$ refer to the location of center (x, y) and width, height of the predicted boxes.

    - $\large x^{\*},\, y^{\*},\, w^{\*},\, h^{\*}$ are for ground-truth boxes.

    - $\large x_{a},\, y_{a},\, w_{a},\, h_{a}$ are for anchor boxes created from anchor generation layers.


<br/>

### **5. Region Proposal Networks (RPN)**

<br/>

```python
class RPN(nn.Module):
    def __init__(self, n_anchor, in_channels, out_channels):
        super(RPN, self).__init__()

        # classifier (whether an anchor contains an object or not)
        self.classifier = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                        nn.Conv2d(out_channels, n_anchor * 2, 1, 1, 0))

        # bounding box regressor (parameterized coordinates)
        self.regressor = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                       nn.Conv2d(out_channels, n_anchor * 4, 1, 1, 0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        class_scores = self.classifier(x)
        anchor_regs = self.regressor(x)

        return class_scores, anchor_regs
```

<br/>

- Generates region proposals along with the objectness scores 

- **Region Proposal**

    - Regressor using Conv2d layer (in_channels = 512, out_channels = 512)
    
    - output size n_batch x (22500 * 4) where 4 represents the 4 parameterized coordinates of each anchor box. 

- **Object Classifier**

    - Classifier using Conv2d layer

    - output size n_batch x (22500 * 2) where 2 represents the objectness scores, each one for positive and negative.


<br/>

### **6. Multi-Task Loss**

<br/>

```python
def multi_task_loss(pred_clss_scores, gt_labels, pred_regs, gt_regs, lamda, use_pos_mask):
    # classification loss
    L_clss = F.cross_entropy(pred_clss_scores, gt_labels.long(), ignore_index=-1)  # normalized with batch size

    pos_idxs = np.where(gt_labels == 1)[0]
    # regression loss is activated only for positive anchors (ground-truth)
    if use_pos_mask:
        pos_mask = torch.zeros_like(pred_regs)
        pos_mask[pos_idxs] = 1
        pred_regs *= pos_mask
        gt_regs *= pos_mask

    # smooth L1
    dreg = torch.abs(pred_regs - gt_regs)
    beta = 1
    L_reg = ((dreg < beta).float() * 0.5 * dreg ** 2) + ((dreg >= beta).float() * (dreg - 0.5))
    N_reg = len(pos_idxs)

    mtl = L_clss + lamda*(L_reg.sum()/N_reg)

    print(f"Classification Error : {L_clss}  Regression Error : {L_reg.sum()/N_reg}")
    return mtl
```

<br/>

&emsp;<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/d90e9c83-ac74-4d65-b14a-a1929aab27d4" width="380">

<br/>

- **L_clss** : classification loss 

    - cross entropy between predicted clss scores and ground truth labels
     

- **L_reg** : regression loss of anchor coordinates

    - L1 smooth loss : activated only for positively labeled anchors (ground-truth labels)
 
    - $\large p_{i}$ denotes predicted probability of anchor $\large i$ being labeled as positive (object)

    - $\large t_{i}$ is a vector containing the 4 parameterized coordinates of the predicted bounding box $\large i$

    - Asterik mark (\*) represents ground-truth.

- **$\large \lambda$** is a weight hyperparameter for balancing the scale between two distinct losses. (L_clss is normalized with batch size and L_reg is normalized with the number of anchors)

    - By default, set as 10.

<br/>

### **7. Proposal Layer : NMS & Proposal Target Layers**

<br/>

- So far, we got region proposals from RPN in previous step and computed the multi-task loss for those proposals. 

- Now, we need to select final target region proposals (Region of Interest, RoI) that are to be passed to subsequent RoI pooling layer.

- To do this, two additional layers 1. NMS and 2. Proposal Target Layers are added.

<br/>

#### **Non Maximum Suppression (NMS)**

<br/>

- This layer aims to remove repetitive anchor boxes that capures identical object. 

- Firstly, get indices descendingly sorted based on the objectness scores computed from RPN layer with pre-defined length (n_pre_nms) and slice the predicted anchor locations and labels with the sorted indices.

- Using while loop, pick the first anchor (the one with largets objectness score) and remove the anchors that have higher IoUs than nms threshold with that selected anchor, assuming that those anchors are capturing same object. 

- Repeat the loop untill the list of ordered indices becomes empty.

<br/>

```python
def non_maximum_supp(pred_regs, objectness_scores, anchor_boxes, img_size, min_size, n_pre_nms, n_post_nms, nms_thresh):
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    wa = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    ctr_ya = anchor_boxes[:, 1] + 0.5 * ha
    ctr_xa = anchor_boxes[:, 0] + 0.5 * wa

    dx = pred_regs[:, 0]
    dy = pred_regs[:, 1]
    dw = pred_regs[:, 2]
    dh = pred_regs[:, 3]

    ctr_x, ctr_y = dx*wa + ctr_xa, dy*ha + ctr_ya
    w, h = torch.exp(dw)*wa, torch.exp(dh)*ha

    roi = torch.zeros_like(pred_regs, dtype=pred_regs.dtype)
    roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3] = ctr_x - w*0.5, ctr_y - h*0.5, ctr_x + w*0.5, ctr_y + h*0.5

    # clipping the min & max of roi to img size
    roi[:, [0, 2]] = torch.clip(roi[:, [0, 2]], 0, img_size[0])
    roi[:, [1, 3]] = torch.clip(roi[:, [1, 3]], 0, img_size[1])

    hs = roi[:, 3] - roi[:, 1]
    ws = roi[:, 2] - roi[:, 0]

    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi, scores = roi[keep], objectness_scores[keep]

    order_idxs = scores.ravel().argsort(descending=True)[:n_pre_nms]
    print(order_idxs)
    roi, scores = roi[order_idxs], scores[order_idxs]

    keep = []
    x1, y1, x2, y2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]
    areas = (x2 - x1 + 1)*(y2 - y1 + 1)    # area of each anchor

    order_idxs = order_idxs.argsort(descending=True)
    while (order_idxs.size(0) > 0):
        i = order_idxs[0]
        keep.append(i)

        max_x1 = torch.maximum(x1[i], x1[order_idxs[1:]])
        max_y1 = torch.maximum(y1[i], y1[order_idxs[1:]])
        min_x2 = torch.minimum(x2[i], x2[order_idxs[1:]])
        min_y2 = torch.minimum(y2[i], y2[order_idxs[1:]])

        inter = torch.maximum(torch.tensor(0.), min_x2 - max_x1 + 1) * torch.maximum(torch.tensor(0.), min_x2 - max_x1 + 1)

        ious = inter / (areas[i] + areas[order_idxs[1:]] - inter)
        keep_idxs = np.where(ious <= nms_thresh)[0]
        order_idxs = order_idxs[keep_idxs+1]

    keep = keep[:n_post_nms]
    roi = roi[keep]

    return roi
```

<br/>

#### **Proposal Target Layer**

<br/>

- Proposal target layer selects samples from region proposals that are mostly useful for traininig the fast R-CNN. 

- Calcuates the IoU between predicted anchors filtered from NMS layer and ground-truth boxes and only choose the anchors with IoU higher than certain threshold (which is here, 0.7) 

- The number of positive sample regions per image is fixed (here, 64) and so is for negative sample regions (64).

<br/>

```python
def proposal_target_layer(roi, bbox, n_sample, pos_ratio, pos_iou_thresh):
    ious = torch.ones_like(roi)*-1

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        ax1, ay1, ax2, ay2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]
        max_x1 = torch.maximum(ax1, torch.tensor(x1))
        max_y1 = torch.maximum(ay1, torch.tensor(y1))
        min_x2 = torch.minimum(ax2, torch.tensor(x2))
        min_y2 = torch.minimum(ay2, torch.tensor(y2))

        idxs = np.where((max_x1 < min_x2) & (max_y1 < min_y2))[0]
        tot_area = (ax2[idxs] - ax1[idxs])*(ay2[idxs] - ay1[idxs]) + (x2-x1)*(y2-y1)
        inter_area = (min_x2[idxs] - max_x1[idxs])*(min_y2[idxs] - max_y1[idxs])
        ious[idxs, i] =  inter_area / (tot_area - inter_area)

    max_iou_gt_idxs = ious.argmax(dim=1)
    max_iou_gt_vals = ious[torch.arange(roi.size(0)), max_iou_gt_idxs]

    # select positive samples
    pos_roi_per_image = n_sample * pos_ratio
    pos_idxs = np.where(max_iou_gt_vals >= pos_iou_thresh)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_idxs.size))

    if len(pos_idxs) > 0:
        pos_idxs = np.random.choice(pos_idxs, size=pos_roi_per_this_image, replace=False)

    # select negtative (backgroud) samples
    neg_idxs = np.where((max_iou_gt_vals < pos_iou_thresh) &
                        (max_iou_gt_vals >= 0))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_idxs.size))

    if neg_idxs.size > 0:
        neg_idxs = np.random.choice(neg_idxs, size = neg_roi_per_this_image, replace=False)

    return ious, pos_idxs, neg_idxs
```

<br/>

- Visualize selected positive and negative achor boxes from proposal target layers onto the image.


&emsp;&emsp; **Positive Anchors**

<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/44db442d-40cc-4486-811b-8cac85fe45cd" width="500">


&emsp;&emsp; **Negative Anchors**

<img src="https://github.com/SuminizZ/Algorithm/assets/92680829/ca594df1-f9c7-489b-8a82-a57196f2ae46" width="500">

<br/>

### **8. Get Ground-Truth Labels and Paramterized Locations of Selected Anchors**

<br/>

- For supervised training in fast R-CNN, we need to prepare ground truth labels and locations of the proposed anchor boxes (RoIs).

<br/>

```python
def ground_truth_rois(ious, rois, bbox, pos_idxs, neg_idxs):
    keep_idxs = torch.Tensor(np.append(pos_idxs, neg_idxs)).long()

    gt_roi_labels = torch.zeros(len(keep_idxs))
    gt_roi_labels[:len(pos_idxs)] = 1

    sroi = rois[keep_idxs]    # sample roi extracted from proposal target layers
    bbox_sroi = bbox[ious.argmax(axis=1)[keep_idxs]]

    eps = torch.finfo(sroi.dtype).eps

    # sampled roi (pred)
    w = torch.maximum(sroi[:, 2] - sroi[:, 0], torch.tensor(eps))
    h = torch.maximum(sroi[:, 3] - sroi[:, 1], torch.tensor(eps))
    ctr_x = sroi[:, 0] + 0.5 * w
    ctr_y = sroi[:, 1] + 0.5 * h

    gt_w = bbox_sroi[:, 2] - bbox_sroi[:, 0]
    gt_h = bbox_sroi[:, 3] - bbox_sroi[:, 1]
    gt_ctr_x = bbox_sroi[:, 0] + 0.5 * gt_w
    gt_ctr_y = bbox_sroi[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ctr_x) / w
    dy = (gt_ctr_y - ctr_y) / h
    dw = torch.log(gt_w / w)
    dh = torch.log(gt_h / h)

    gt_roi_regs = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dw.unsqueeze(1), dh.unsqueeze(1)], dim=1)

    return gt_roi_labels, gt_roi_regs
```

<br/>

### **9. RoI Pooling**

<br/>

- Through this layer, actual features of the image (from feature extractor layer) now will be pooled based on the previously gained labels and coordinates of target RoIs.

- This layer will adopt **global average pooling layer** to get final feature extractions, which are the inputs of the fast R-CNN. 

<br/>

```python
# get img feature maps at predicted coordinates

def roi_pooling(rois, pos_idxs, neg_idxs, output_map):
    keep_idxs = np.append(pos_idxs, neg_idxs)
    gt_roi_labels = torch.ones(len(keep_idxs))
    gt_roi_labels[len(pos_idxs):] = 0
    sample_rois = rois[keep_idxs]

    admaxpool = nn.AdaptiveMaxPool2d((7,7), True)

    output = []
    rois = sample_rois*(1/16)    # sub-sampling ratio
    rois = rois[:, [1, 0, 3, 2]]
    rois = rois.long()    # turn each loc into integers

    for i in range(len(rois)):
        roi = rois[i]
        im = output_map.narrow(0, 0, 1)[..., roi[0]:(roi[2]+1), roi[1]:(roi[3]+1)]   # i : channel indexing
        tmp = admaxpool(im)
        output.append(tmp[0])

    output = torch.cat(output, 0)     # 128 x 512 x 7 x 7
    print(f"roi pooling output size : {output.size()}")
    k = output.view(output.size(0), -1)   # 128 x (512*7*7 = 25088)

    return k
```

<br/>

- For all those positive (64) and negative roi (64) samples, we need to multiply the sub-sampling ratio (1/16) to rescale the locations of roi (800 x 800) into feature map size (50 x 50). 

- Then performs roi pooling across all samples and apply adaptive GAP to make final output feature map size 7 x 7.  

- After completing the for loop (x 128), concat the resultant (512 x 7 x 7) feature maps along the dimension 0. (final output shape : 128 x 512 x 7 x 7)

<br/>

### **10. Fast R-CNN**

<br/>

- The last fully connected layers of this model that performs classification and regression tasks to get final labels and locations of pooled RoIs. 

<br/>

```python
class FastRCNN(nn.Module):
    def __init__(self, k, out_channels):
        super(FastRCNN, self).__init__()
        num_rois, in_channels = k.shape[0], k.shape[1]
        self.base = nn.Sequential(nn.Linear(in_channels, out_channels),
                                  nn.Linear(out_channels, out_channels//2))
        self.roi_regs = nn.Linear(out_channels//2, 8)    # 8 : 1 class, 1 background, 4 coordiinates
        self.roi_clss = nn.Linear(out_channels//2, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        pred_roi_regs = self.roi_regs(x)
        pred_roi_clss = self.roi_clss(x)

        return pred_roi_regs, pred_roi_clss
```

<br/>

- Input : flattened features extracted from RoI pooling layer (128 x (512*7*7 = 25088))

- Ouput : 

    - Regression : coordinates of background (if labeled as 0) and object (if label = 1)

    - Classification : assign label 0 or 1 to tell if the given anchor captures an object or not. 


- Finally, calculate multi task loss between predicted anchors and ground truth boxes. 

<br/>

```python
pred_roi_regs = pred_roi_regs.view(n_sample, -1, 4)
pred_roi_regs = pred_roi_regs[torch.arange(0, n_sample).long(), gt_roi_labels.long()]
mtl = multi_task_loss(pred_roi_clss, gt_roi_labels, pred_roi_regs, gt_roi_regs, lamda, False)
print(f"Multi Task Loss : {mtl}")
```

<br/>

```
Classification Error : 0.6667768955230713  Regression Error : 2.890517234802246
Multi Task Loss : 29.571949005126953
```
<br/>


--- 

<br/>

- So far, we've implemented Faster R-CNN with PyTorch. 

- Faster R-CNN recorded 75.9 % of mAP when trained on COCO + PASCAL VOC 2007 + PASCAL VOC 2012, outperforming previous selective search based model. 

&emsp;&emsp; <img src="https://github.com/SuminizZ/Algorithm/assets/92680829/7fa0024c-c4fb-4cd2-bc82-9e965bd7a4a8" width="750">

<br/>

- Next post will cover **YOLO v1** that uses single-stage object detection, integrating RPN and classifier into a single CNN networks to process the entire image at once.