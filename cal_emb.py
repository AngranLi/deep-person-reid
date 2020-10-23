import cv2
import os
import torch
import torchreid
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import models


def load_pretrained_model(sd_path):

    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=751,
        loss='softmax',
        pretrained=False
    )
    model = model.cuda()

    state_dict = torch.load(sd_path)
    model_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    model = model.eval()

    return model

def test_pretrained_model(model):

    datamanager = torchreid.data.ImageDataManager(
        root='/home/angran/GIT/deep-person-reid/reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    # Build engine
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    # Run training and test
    engine.run(
        save_dir='log/osnet',
        test_only=True,
        dist_metric='cosine'
    )

    return


def person_detection(img):
    img_pt = F.to_tensor(img).unsqueeze(0)
    img_pt = img_pt.to('cuda:0')

    kprcnn = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()
    kprcnn = kprcnn.to('cuda:0')

    with torch.no_grad():
        output = kprcnn(img_pt)[0]

    bbps = []
    kpts = []
    for i in range(output['scores'].cpu().numpy().shape[0]):
        if output['scores'][i].cpu().numpy() > 0.9:
            bbps.append(output['boxes'][i].cpu().numpy().astype(int))
            kpts.append(output['keypoints'][i].cpu().numpy().astype(int)[:, :2])
    
    return bbps, kpts

def generate_embs(root_dir):

    for root, dirs, files in os.walk(root_dir):
        print(root)
        for fn in files:
            if fn.endswith('.jpg'):
                img = Image.open(os.path.join(root, fn))
                bbps, kpts = person_detection(img)
                embs = []
                for bbp in bbps:
                    person = img.crop(bbp)
                    person = F.to_tensor(person).unsqueeze(0)
                    # A trick needs to be considered.
                    if person.shape[-1] * person.shape[-2] < 1600:
                        continue
                    person = person.to('cuda:0')
                    with torch.no_grad():
                        emb = model(person)
                    embs.append(emb.cpu().numpy())
                
                np.save(os.path.join(root, fn[:-4]), np.array(embs).squeeze())
    
    return



if __name__ == '__main__':

    sd_path = '/home/angran/GIT/deep-person-reid/log/osnet/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
    model = load_pretrained_model(sd_path)

    # test_pretrained_model(model)
    
    root_dirs = [
        '/nasty/scratch/common/msg/tms/Gen-1.1-6ft/Mt-Healthy/',
        '/nasty/scratch/common/msg/tms-gen1/reds/gen1_3imgs_perwt/'
        # '/nasty/scratch/common/msg/tms/Gen-1.1-6ft/Mt-Healthy/frames/alert-data-2020-10-08/ALR-1602085237421-574132'
    ]
    for root_dir in root_dirs:
        generate_embs(root_dir)
    
                