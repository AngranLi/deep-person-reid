import cv2
import os
import torch
import torchreid
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision import models


class Person():
    
    def __init__(self, num=None, bounding_box=[], prob=0, file_name='', embedding=0):
        self.id   = num
        self.bbox = bounding_box
        self.prob = prob
        self.fn   = file_name
        self.emb  = embedding


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


def get_embs(model, predictions, img_names, imgs):

    people = []
    for n in range(len(predictions)):
        for i in range(predictions[n]['scores'].cpu().numpy().shape[0]):
            if predictions[n]['scores'][i].cpu().numpy() > 0.9:
                bbp = predictions[n]['boxes'][i].cpu().numpy().astype(int)
                if (bbp[3] - bbp[1]) * (bbp[2] - bbp[0]) < 1600: # 10000?
                    continue
                prob = predictions[n]['scores'][i].cpu().numpy()
                imp = imgs[n].copy()
                imp = imp[bbp[1]:bbp[3], bbp[0]:bbp[2]]
                imp = F.to_tensor(imp).to('cuda:0').unsqueeze(0)
                
                with torch.no_grad():
                    emb = model(imp)
                emb = torch.squeeze(emb)
                n_emb = emb / emb.norm()
                tmp_person = Person(num=i, bounding_box=bbp, prob=prob, file_name=img_names[n], embedding=n_emb)
                people.append(tmp_person)
    
    return people


def get_imgs(root_dir):

    imgs = []
    imgs_pt = []
    img_names = []

    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith('.jpg'):
                img_names.append(fn)
                img = plt.imread(os.path.join(root, fn))
                imgs.append(img)
                imgs_pt.append(F.to_tensor(img.copy()).to('cuda:0'))

    return imgs, imgs_pt, img_names


def people_of_img(people, file_name):
    """ Get the list of Person() of people in this image.
    """
    
    ppl_lst = []
    for person in people:
        if person.fn == file_name:
            ppl_lst.append(person)
            
    return ppl_lst


def img_with_most_ppl(people, img_names):
    """ Get the image name with most people in it.
    """

    pers_num = []
    for img_name in img_names:
        pers_num.append(len(people_of_img(people, img_name)))
    
    return img_names[np.argmax(pers_num)]


def set_person_id(people, img_names, img_most_ppl):
    """
    """

    img_names.remove(img_most_ppl)
    ppl_most = people_of_img(people, img_most_ppl)
    for img_name in img_names:  
        ppl_look_from = people_of_img(people, img_name)
        for person in ppl_look_from:
            cos_dists = {}
            for pers_look_for in ppl_most:
                cos_dists[(torch.dot(person.emb, pers_look_for.emb).item())] = pers_look_for.id
            person.id = cos_dists[max(list(cos_dists.keys()))]


def plot_for_test(people, root_dir, img_names):

    for fn in img_names:
        im = plt.imread(os.path.join(root_dir, fn))
        people_here = people_of_img(people, fn)
        for person in people_here:
            label = str(person.id)
            cv2.rectangle(im, tuple(person.bbox[:2]), tuple(person.bbox[2:]), color=(0, 255, 0))
            cv2.putText(im,label,tuple(person.bbox[:2]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
        fig, ax = plt.subplots(figsize = (16,9))
        ax.imshow(im)
        plt.savefig(os.path.join(root_dir, 'reid_' + fn))
        plt.close(fig)

if __name__ == '__main__':

    sd_path = '/home/angran/GIT/deep-person-reid/log/osnet/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
    osnet = load_pretrained_model(sd_path)

    # test_pretrained_model(osnet)

    kprcnn = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    kprcnn.eval()
    kprcnn = kprcnn.to('cuda:0')
    
    root_dirs = [
        '/nasty/scratch/common/msg/tms/Gen-1.1-6ft/Mt-Healthy/',
        '/nasty/scratch/common/msg/tms-gen1/reds/gen1_3imgs_perwt/'
    ]
    for root_dir in root_dirs:
        old_root = ''
        for root, dirs, files in os.walk(root_dir):
            for fn in files:
                if fn.endswith('.npy') or fn.endswith('.pickle'):
                    os.remove(os.path.join(root, fn))

                if fn.endswith('.jpg'):
                    new_root = root
                    if old_root != new_root:
                        print(root)
                        old_root = new_root
                        imgs, imgs_pt, img_names = get_imgs(old_root)

                        # Person Detection
                        with torch.no_grad():
                            predictions = kprcnn(imgs_pt)

                        # list of Person()
                        people = get_embs(osnet, predictions, img_names, imgs)

                        # Get the image name with most people in it.
                        img_most_ppl = img_with_most_ppl(people, img_names)
                        
                        # Set the person id in each image according to the person id 
                        # in the image with most people in it
                        set_person_id(people, img_names.copy(), img_most_ppl)
                        
                        plot_for_test(people, old_root, img_names)