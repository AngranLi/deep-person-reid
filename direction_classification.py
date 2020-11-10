import cv2
import os, glob
import torch
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation
from pathlib import Path
from PIL import Image
from tritonclient import grpc as triton
from torchvision import transforms, models


class DetectionPipeline:
    """Pipeline class for detecting people and faces in the frames of a video file."""
    
    def __init__(self, triton_host, n_frames=None, batch_size=16, resize=(1280, 800)):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with detector. (default: {32})
            resize {tuple} -- Dimensions to resize frames to before inference. (default: {(1280, 800)})
        """
        self.triton_client = triton.InferenceServerClient(url=triton_host)
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def detector(self, frames):
        infer_inputs = [
            triton.InferInput('input_1', (len(frames), 3, *self.resize[::-1]), "FP32")
        ]
        frames = np.array(frames, dtype=np.float32)
        frames = np.transpose(frames, (0, 3, 1, 2))
        infer_inputs[0].set_data_from_numpy(frames)
        result = self.triton_client.infer('retinanet', infer_inputs)
        scores = result.as_numpy('scores').reshape((-1, 100))
        boxes = result.as_numpy('boxes').reshape((-1, 100, 4))
        classes = result.as_numpy('classes').reshape((-1, 100))
        
        return scores, boxes, classes
    
    def __call__(self, filename=None, fps=None):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
            fps{list of str} -- File paths to images
        """
        
        ## Video
        if filename is not None:
            # Create video reader and find length
            v_cap = cv2.VideoCapture(filename)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"frame count: {v_len}")
            if v_len > 2400 or v_len <= 0:
                print(f"Video broken.")
                return
            v_fps = int(v_cap.get(cv2.CAP_PROP_FPS))

            if v_len >= v_fps * 4:
                idx_0 = v_len//2 - v_fps * 2 # 1 second before triggered time
                idx_m = v_len//2 - v_fps     # approx. 1.0s latency for the camera
                idx_l = v_len//2             # 1 second after triggered time
            else: 
                idx_0 = max(0, v_len//2 - v_fps)
                idx_m = v_len//2
                idx_l = min(v_len-1, v_len//2 + v_fps)

            # Pick 'n_frames' evenly spaced frames to sample
            if self.n_frames is None:
                sample = np.arange(idx_0, idx_l)
            else:
                sample = np.linspace(idx_0, idx_l, self.n_frames).astype(int)
            
            length = v_len
            
        ## Images
        elif fps is not None:
            sample = np.linspace(0, len(fps)-1, self.n_frames).astype(int)

            length = len(fps)
    
        # Loop through frames
        scores = []
        boxes = []
        classes = []
        frames = []
        frames_all = []
        
        for j in range(length):
            if j in sample:
                # Load frame
                ## Video
                if filename is not None:
                    success = v_cap.grab()
                    success, frame = v_cap.retrieve()
                    print(type(frame))
                    if not success:
                        continue
                ## Images
                elif fps is not None:
                    frame = cv2.imread(fps[j])
                    
                # Resize frame to desired size and reorder channels
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)
                frames_all.append(frame)
                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    scores_batch, boxes_batch, classes_batch = self.detector(frames)
                    scores.extend(scores_batch)
                    boxes.extend(boxes_batch)
                    classes.extend(classes_batch)
                    frames = []
        
        if filename is not None:
            v_cap.release()
            

        return frames_all, np.array(scores), np.array(boxes), np.array(classes)


def load_pretrained_model(sd_path):

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 3)
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


def iou(bb1, bb2):
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    
    xx1 = np.maximum(bb1[0], bb2[0])
    yy1 = np.maximum(bb1[1], bb2[1])
    xx2 = np.minimum(bb1[2], bb2[2])
    yy2 = np.minimum(bb1[3], bb2[3])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)

    inter = w * h
    iou = inter / (area1 + area2 - inter)
    return iou

class Person:
    
    def __init__(self, num, box, prob, frame_i, embedding=0, direc=[255, 255, 0]):
        """ 
        Args:
            num (int): id of this person.
            box (list of int): left/top/right/bottom boundary of bounding box.
            prob (float, range[0, 1]): score for person detection.
            frame_i (int): frame number in which this person exists.
            direc (list of int, range[0, 255]): red-away, yellow-other, green-toward.
        """
        self.id = num
        self.boxes = [box]
        self.box_sz = [(box[3] - box[1]) * (box[2] - box[0])]
        self.probs = [prob]
        self.frames_i = [frame_i]
        self.centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]
        self.emb  = [embedding]
        self.direc = direc
    
    
    def add_frame(self, box, prob, frame_i, embedding):
        self.boxes.append(box)
        self.box_sz.append([(box[3] - box[1]) * (box[2] - box[0])])
        self.probs.append(prob)
        self.frames_i.append(frame_i)
        self.centers.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        self.emb.append(embedding)
    
    def compare_iou(self, bounding_box):
        return iou(self.boxes[-1], bounding_box)
    
    def compare_emb(self, embedding):
        return np.dot(self.emb[-1], embedding)


def plot_boxes(img, boxes, classes):
#     cmap = cm.get_cmap('jet')
#     colors = [cmap(c / max(classes))[:3] for c in classes]
#     colors = [[int(c * 255) for c in v] for v in colors]
    boxes = boxes.astype(int)

    img_plt = img.copy()
#     for box, col in zip(boxes, colors):
    for box in boxes:
#         img_plt = cv2.rectangle(img_plt, tuple(box[:2]), tuple(box[2:]), col, 3)
        img_plt = cv2.rectangle(img_plt, tuple(box[:2]), tuple(box[2:]), [0, 0, 255], 3)

    return img_plt


def plot_directions(img, boxes, directions):
    img_plt = img.copy()
    for box, direc in zip(boxes, directions):
        img_plt = cv2.rectangle(img_plt, tuple(box[:2]), tuple(box[2:]), direc, 3)

    return img_plt


def people_of_img(frames, scores, boxes, classes, model):

    people = []
    for i in range(len(frames)):
        mask = (scores[i] > 0.6) & (classes[i] == 0)
        scores_i = scores[i, mask]
        boxes_i = boxes[i, mask]
        
        # Discard overlapped bboxes for same subject
        m = 0
        n = 1
        while m < len(boxes_i) - 1:
            while n < len(boxes_i):
                if iou(boxes_i[m], boxes_i[n]) > 0.4:
                    if scores_i[m] > scores_i[n]:
                        scores_i = np.delete(scores_i, n, 0)
                        boxes_i = np.delete(boxes_i, n, 0)
                    else:
                        scores_i = np.delete(scores_i, m, 0)
                        boxes_i = np.delete(boxes_i, m, 0)
                n += 1
            m += 1
            n = m + 1
        
        # Calculate embeddings for all the detected subjects
        img = frames[i].copy()

        embs = []
        boxes_i = boxes_i.astype(int)
        for j in range(len(scores_i)):
            imp = img[boxes_i[j, 1]:boxes_i[j, 3], boxes_i[j, 0]:boxes_i[j, 2]]
            imp = transforms.functional.to_tensor(imp).to('cuda:0').unsqueeze(0)
            with torch.no_grad():
                emb = model(imp)
            emb = torch.squeeze(emb)
            emb = emb / emb.norm()
            embs.append(emb.cpu().numpy())
        
        # Include all people in the first image
        if len(people) == 0:
            for j in range(len(scores_i)):
                people.append(Person(j, boxes_i[j], scores_i[j], i, embs[j]))
            continue
            
        # Find corresponding person according to emb
        for j in range(len(scores_i)):
            cos_dist = []
            for person in people:
                cos_dist.append(person.compare_emb(embs[j]))
            print(f"cos_dist: {cos_dist}")
            min_dist_ind = np.argmin(cos_dist)
            if cos_dist[min_dist_ind] < 0.5: # Found the corresponding person
                people[min_dist_ind].add_frame(boxes_i[j], scores_i[j], i, embs[j])
            else: # Add the subject as a new person
                people.append(Person(len(people), boxes_i[j], scores_i[j], i, embs[j]))

        '''
        # Find corresponding person according to IOU
        for j in range(len(scores_i)):
            ious_ij = []
            for person in people:
                ious_ij.append(person.compare_iou(boxes_i[j]))
            max_iou_ind = np.argmax(ious_ij)
            if ious_ij[max_iou_ind] > 0.4: # Found the corresponding person
                people[max_iou_ind].add_frame(boxes_i[j], scores_i[j], i)
            else: # Add the subject as a new person
                people.append(Person(len(people), boxes_i[j], scores_i[j], i))
        '''

    # Replace this mess with actual gate coordinates
    people = [p for p in people if np.abs(np.array(p.centers)[:, 0] - 600).min() < 250]
    people = [p for p in people if np.mean(p.box_sz) > 10000]

    for person in people.copy():
        if len(person.boxes) <= 1:
            people.remove(person)

    return people


def cal_movement(people, frames, orient_cls):

    for person in people:
        person_imgs = []
        for i in person.frames_i:
            bb = person.boxes[person.frames_i.index(i)].astype(int)
            person_img = Image.fromarray(frames[i][bb[1]:bb[3], bb[0]:bb[2]])
            person_img = transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.ToTensor()
            ])(person_img)
            person_imgs.append(person_img)
        person_imgs = torch.stack(person_imgs).to('cuda:0')
        logits = orient_cls(person_imgs)
        probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        direcs = list(np.argmax(probs, axis=1))
        direc = max(direcs, key=direcs.count)
        if direc == 0:
            person.direc = [0, 255, 0]
        elif direc == 1:
            person.direc = [255, 255, 0]
        elif direc == 2:
            person.direc = [255, 0, 0]
        else:
            print(f"Orientation classification failed, the direcs list is {direcs}.")
            return None
    
    return people
    '''
    for person in people:
        bb = np.array(person.boxes)
        top = bb[:, 1]
        bot = bb[:, 3]
        if list(800 - bot < 50).count(True) > 5:
            print("top")
            x = np.arange(len(top))
            k, b = np.polyfit(x, top, 1)
        else:
            x = np.arange(len(bot))
            k, b = np.polyfit(x, bot, 1)

        if k > 4:
            person.direc = [255, 0, 0]
        elif k < -4:
            person.direc = [0, 255, 0]
        else:
            person.direc = [255, 255, 0]
    '''

    return people


def save_imgs(people, frame_i, fn):

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    img_num = 0
    for i in frame_i:
        ax = axs[img_num//2][img_num%2]
        box_in_img = []
        direc_in_img = []
        for person in people:
            try:
                box_in_img.append(person.boxes[person.frames_i.index(i)])
                direc_in_img.append(person.direc)
            except Exception as e:
                # print(f"Got error: {e}")
                pass
                    
        ax.imshow(plot_directions(frames[i], box_in_img, direc_in_img))
        ax.set_title(f"frame {i}")
        img_num += 1
    
    fp = '/'.join(fn.split('/')[:-1])
    Path(fp).mkdir(parents=True, exist_ok=True)
    fig.savefig(fn + '.png')
    plt.close()


if __name__ == '__main__':

    # fps = glob.glob('/nasty/data/msg-ml/data/mt-healthy/tms/6ft/tms-1-and-2/**/*.mkv', recursive=True)
    # detector = DetectionPipeline('10.8.8.210:8001', 15)
    # img_save_path = '/nasty/scratch/common/msg/tms/Gen-1.1-6ft/Mt-Healthy/direction_classification'  

    # idx_lst = []
    # for _ in range(200):
    #     idx = np.random.randint(0, len(fps))
    #     if idx in idx_lst:
    #         continue
    #     idx_lst.append(idx)
    #     print(f"idx: {idx}")
    #     try:
    #         frames, scores, boxes, classes = detector(fps[idx])
    #     except Exception as e:
    #         print(f"Video {idx} is broken.")
    #         continue
    
    detector = DetectionPipeline('10.8.8.210:8001', 15)

    sd_path = '/home/angran/GIT/deep-person-reid/log/osnet/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
    osnet = load_pretrained_model(sd_path)

    static_dict_path = '/home/angran/GIT/jupyter/logs/rn_orient/best.pt'
    orient_cls = load_pretrained_model(static_dict_path)

    img_save_path = '/nasty/scratch/common/msg/tms-gen1/reds/direc_cls_rn_1'

    tmp_roots = glob.glob('/nasty/scratch/common/msg/tms-gen1/reds/gen1/**/*.jpeg', recursive=True)
    roots = []
    for root in tmp_roots:
        if '@' not in root:
            roots.append('/'.join(root.split('/')[:-1]))
    roots = np.unique(roots)
    del tmp_roots
    
    idx_lst = []
    for _ in range(100):
        idx = np.random.randint(0, len(roots))
        if idx in idx_lst:
            continue
        idx_lst.append(idx)
        print(f"idx: {idx}")

        root = roots[idx]
        fps = sorted(glob.glob(root + '/*.jpeg', recursive=True))

        frames, scores, boxes, classes = detector(fps=fps)

        people = people_of_img(frames, scores, boxes, classes, osnet)
        print(f"num of people between pillars: {len(people)}")

        # people = cal_movement(people, frames, orient_cls)
 
        # fn = os.path.join(img_save_path, '/'.join(root.split('/')[-3:]))
        # frame_i = [0, len(frames)//3, len(frames)//3 * 2, len(frames)-1]
        # save_imgs(people, frame_i, fn)
