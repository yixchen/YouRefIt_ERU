# YouRefIt: Embodied Reference Understanding with Language and Gesture
[YouRefIt: Embodied Reference Understanding with Language and Gesture](https://yixchen.github.io/YouRefIt/)

by [Yixin Chen](https://yixchen.github.io/), [Qing Li](https://liqing-ustc.github.io/), [Deqian Kong](https://sites.google.com/view/deqiankong/home), [Yik Lun Kei](https://allenkei.github.io/), [Tao Gao](http://www.stat.ucla.edu/~taogao/Taogao.html), [Yixin Zhu](http://www.yzhu.io/), [Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/) and [Siyuan Huang](http://www.siyuanhuang.com/)

The IEEE International Conference on Computer Vision (ICCV), 2021


## Introduction
We study the machine's understanding of embodied reference: One agent uses both language and gesture to refer to an object to another agent in a shared physical environment. To tackle this problem, we introduce YouRefIt, a new crowd-sourced, real-world dataset of embodied reference.

For more details, please refer to our [paper](https://yixchen.github.io/YouRefIt/file/iccv2021_yourefit.pdf).

<p align="center">
  <img src="teaser/overview.png" width="50%"/>
</p>

## Checklist

+ [x] Image ERU
+ [x] Video ERU dataset

## Installation

The code was tested with the following environment: Ubuntu 18.04/20.04, python 3.7/3.8, pytorch 1.9.1.
Run
```bash
    git clone https://github.com/yixchen/YouRefIt_ERU
    pip install -r requirements.txt
```

## Dataset
Download the YouRefIt dataset from [Dataset Request Page](https://yixchen.github.io/YouRefIt/request.html) and put under ```./ln_data```

## Model weights
* [Yolov3](https://pjreddie.com/media/files/yolov3.weights): download the pretrained model and place the file in ``./saved_models`` by 
    ```
    sh saved_models/yolov3_weights.sh
    ```
* More pretrained models are availble [Google drive](https://drive.google.com/drive/folders/1jphUV6if1ka3LyCxXMPbNB65n26Y7nNH?usp=sharing), and should also be placed in ``./saved_models``.

Make sure to put the files in the following structure:

```
|-- ROOT
|	|-- ln_data
|		|-- yourefit
|			|-- images
|			|-- paf
|			|-- saliency
|	|-- saved_modeks
|		|-- final_model_full.tar
|		|-- final_resc.tar
```

## Training
Train the model, run the code under main folder. 

    python train.py --data_root ./ln_data/ --dataset yourefit --gpu gpu_id 

## Evaluation
Evaluate the model, run the code under main folder. 
Using flag ``--test`` to access test mode.

    python train.py --data_root ./ln_data/ --dataset yourefit --gpu gpu_id \
     --resume saved_models/model.pth.tar \
     --test

### Evaluate Image ERU on our released model
Evaluate our full model with PAF and saliency feature, run 

    python train.py --data_root ./ln_data/ --dataset yourefit  --gpu gpu_id \
     --resume saved_models/final_model_full.tar --use_paf --use_sal --large --test

Evaluate baseline model that only takes images as input, run 

    python train.py --data_root ./ln_data/ --dataset yourefit  --gpu gpu_id \
     --resume saved_models/final_resc.tar --large --test

Evalute the inference results on test set on different IOU levels by changing the path accordingly,

     python evaluate_results.py

### Citation

    @inProceedings{chen2021yourefit,
     title={YouRefIt: Embodied Reference Understanding with Language and Gesture},
     author = {Chen, Yixin and Li, Qing and Kong, Deqian and Kei, Yik Lun and Zhu, Song-Chun and Gao, Tao and Zhu, Yixin and Huang, Siyuan},
     booktitle={The IEEE International Conference on Computer Vision (ICCV),
     year={2021}
     }    

### Acknowledgement
Our code is built on [ReSC](https://github.com/zyang-ur/ReSC) and we thank the authors for their hard work.
