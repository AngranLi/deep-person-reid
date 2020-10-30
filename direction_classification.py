import cv2
import os, glob
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation
from pathlib import Path
from PIL import Image
from tritonclient import grpc as triton


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
    
    def __init__(self, num, box, prob, frame, direc=[255, 255, 0] ):
        self.id = num
        self.boxes = [box]
        self.box_sz = [(box[3] - box[1]) * (box[2] - box[0])]
        self.probs = [prob]
        self.frames = [frame]
        self.centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]
        self.direc = direc
    
    
    def add_frame(self, box, prob, frame):
        self.boxes.append(box)
        self.box_sz.append([(box[3] - box[1]) * (box[2] - box[0])])
        self.probs.append(prob)
        self.frames.append(frame)
        self.centers.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    
    def compare(self, bounding_box):
        return iou(self.boxes[-1], bounding_box)


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


def people_of_img(frames, scores, boxes, classes):

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
        
        if len(people) == 0:
            for j in range(len(scores_i)):
                people.append(Person(j, boxes_i[j], scores_i[j], i))
            continue
        
        for j in range(len(scores_i)):
            ious_ij = []
            for person in people:
                ious_ij.append(person.compare(boxes_i[j]))
            max_iou_ind = np.argmax(ious_ij)
            if ious_ij[max_iou_ind] > 0.4:
                people[max_iou_ind].add_frame(boxes_i[j], scores_i[j], i)
            else:
                people.append(Person(len(people), boxes_i[j], scores_i[j], i))

    # Replace this mess with actual gate coordinates
    people = [p for p in people if np.abs(np.array(p.centers)[:, 0] - 600).min() < 250]
    people = [p for p in people if np.mean(p.box_sz) > 10000]

    for person in people.copy():
        if len(person.boxes) <= 1:
            people.remove(person)

    return people


def cal_movement(people):

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
                box_in_img.append(person.boxes[person.frames.index(i)])
                direc_in_img.append(person.direc)
            except:
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
    img_save_path = '/nasty/scratch/common/msg/tms-gen1/reds/direc_cls'
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


        people = people_of_img(frames, scores, boxes, classes)
        print(f"num of people between pillars: {len(people)}")

        people = cal_movement(people)
 
        fn = os.path.join(img_save_path, '/'.join(root.split('/')[-3:]))
        frame_i = [0, len(frames)//3, len(frames)//3 * 2, len(frames)-1]
        save_imgs(people, frame_i, fn)