import pickle
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

# import glob
# feat_path = '/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/'
# word = pickle.load(open('/home/lipin/code/DDPN/refcoco/format_train.pkl', 'rb'), encoding='iso-8859-1')
'''
'4223_2':  {'iid': 533619, 'boxes': [array([   0.79,   88.83,  219.8 ,  344.08])], 'img_path': '/data/chenchao/work/visualground/visualground_caffe/data/refer/images/mscoco/image2014/train2014/COCO_train2014_000000533619.jpg', 'qstr': 'woman on left'}
'''
'''
print(word.keys())
print(word['4223_2'])

data_list = []
for name in word.keys():
    data_list.append(word[name])
print(data_list.__len__())

# '/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/COCO_train2014_000000210642.jpg.npz'
# print(glob.glob(feat_path + '*.npz'))

iid = data_list[0]['iid']
print(iid)

all_file_in_feat_path = glob.glob(feat_path + '*')

print('/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/COCO_train2014_' + '000000000000'[0:12-str(iid).__len__()] + str(iid) + '.jpg.npz')

r = np.load('/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/COCO_train2014_000000210642.jpg.npz')
'''
def get_img_feat(feat_path):  # a image has 100 box,has 100 imge_feat(2048dim)
    r = np.load(feat_path)

    x = r['x']

    print('img_feat shape', x.shape)
    print('img_feat[:, 0]', x[:, 0])  # 2048*100,maybe 2048*89,error comes up

    x_transpose = np.transpose(x)
    print('img_feat shape after transpose', x_transpose.shape)

    img_feat = np.zeros((100, 2048))

    if x_transpose.shape[0] > 100:  # maybe >100 or <100
        img_feat = x_transpose[0:100, :]
    if x_transpose.shape[0] <= 100:
        img_feat[0:x_transpose.shape[0], :] = x_transpose

    return img_feat

def get_spt_feat(feat_path):  # (feat_path here is of a image)a image has 100 bbox,has 100 spt_feat

    r = np.load(feat_path)

    # x = r['x']  # 100*2048


    bbox = r['bbox']  # 100*4
    image_h = r['image_h']  # 480
    image_w = r['image_w']  # 640

    spt_feat = np.zeros((100, 5))

    if bbox.shape[0] > 100:
        spt_feat[:, 0] = bbox[0:100, 0] / float(image_w)
        spt_feat[:, 1] = bbox[0:100, 1] / float(image_h)
        spt_feat[:, 2] = bbox[0:100, 2] / float(image_w)
        spt_feat[:, 3] = bbox[0:100, 3] / float(image_h)

        bbox_w = bbox[:, 2] - bbox[0:100, 0] + 1
        bbox_h = bbox[:, 3] - bbox[0:100, 1] + 1

        spt_feat[:, 4] = (bbox_w * bbox_h) / (image_w * image_h)

    if bbox.shape[0] <= 100:
        spt_feat[0:bbox.shape[0], 0] = bbox[:, 0] / float(image_w)
        spt_feat[0:bbox.shape[0], 1] = bbox[:, 1] / float(image_h)
        spt_feat[0:bbox.shape[0], 2] = bbox[:, 2] / float(image_w)
        spt_feat[0:bbox.shape[0], 3] = bbox[:, 3] / float(image_h)

        bbox_w = bbox[0:bbox.shape[0], 2] - bbox[:, 0] + 1
        bbox_h = bbox[0:bbox.shape[0], 3] - bbox[:, 1] + 1

        spt_feat[0:bbox.shape[0], 4] = (bbox_w * bbox_h) / (image_w * image_h)

        '''
        
            spt_feat[:, 0] = bbox[:, 0] / float(image_w)
            spt_feat[:, 1] = bbox[:, 1] / float(image_h)
            spt_feat[:, 2] = bbox[:, 2] / float(image_w)
            spt_feat[:, 3] = bbox[:, 3] / float(image_h)
        
        
            bbox_w = bbox[:, 2]-bbox[:, 0]+1
            bbox_h = bbox[:, 3]-bbox[:, 1]+1
        
            spt_feat[:, 4] = (bbox_w*bbox_h)/(image_w*image_h)
        '''

    return spt_feat

def get_p(format_path):
    #format_path = '/home/lipin/code/DDPN/refcoco/format_train.pkl'
    f = pickle.load(open(format_path, 'rb'), encoding='iso-8859-1')
    anno_list = []
    for name in f.keys():
        anno_list.append(f[name])
    #print(anno_list[0]['qstr'])
    #return anno_list[0]['qstr']
    return anno_list[0]['qstr']

def str2list(words):
    words_list = words.split()

    print('words_list', words_list)

    return words_list

QUERY_MAXLEN = 15
def lookup_dict(dict1, words_list):
    idxes = np.zeros(15).astype(int)  # 0:<blank>


    if len(words_list) > 15:
        words_list = words_list[0:15]

    i = 0

    for word in words_list:
        idxes[i] = dict1.get(word)
        i += 1
    print('idxes:', idxes)
    # idxes1 = dict1.get(word for word in words_list)
    # print('idxes1:', idxes1)
    print(idxes[0])

    return idxes

BATCHSIZE = 64
FEAT_DIR = '/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/'
COCO_PREFIX = 'COCO_train2014_'
FEAT_SUFFIX = '.jpg.npz'

ANNO_DIR = '/home/lipin/code/DDPN/refcoco/'

def net_input(format_anno_list,dict1):  # a batchsize data feed into the net

    # format_train_data_dict = pickle.load(open('/home/lipin/code/DDPN/refcoco/format_train.pkl', 'rb'), encoding='iso-8859-1')
    #
    # format_anno_list = []
    # for name in format_train_data_dict.keys():
    #     format_anno_list.append(format_train_data_dict[name])
    #
    # len = format_anno_list.__len__()
    #
    # print('len', len)

    img_feat = np.zeros((BATCHSIZE, 100, 2048))
    spt_feat = np.zeros((BATCHSIZE, 100, 5))
    p = np.zeros((BATCHSIZE, 15))
    gt_boxes = np.zeros((BATCHSIZE, 4))
    i = 0

    for a_anno in format_anno_list:

        iid = a_anno['iid']

        feat_path = '/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/COCO_train2014_'+'000000000000'[0:12-str(iid).__len__()]+str(iid)+'.jpg.npz'

        words = str2list(a_anno['qstr'])

        gt_boxes[i, :] = np.array(a_anno['boxes'])
        print(get_img_feat(feat_path).shape)
        img_feat[i, :, :] = get_img_feat(feat_path)
        spt_feat[i, :, :] = get_spt_feat(feat_path)

        p[i, :] = lookup_dict(dict1, words)

        i += 1

    print('i:', i)  # see if i equals len

    # get image extracted feat
    '''img_feat = np.zeros((BATCHSIZE, 100, 2048))
    spt_feat = np.zeros((BATCHSIZE, 100, 5))
    p = np.zeros((BATCHSIZE, 15))

    i = 0
    for feat_file_path in glob.glob(FEAT_DIR+'*.npz'):
        img_feat[i, :, :] = get_img_feat(feat_file_path)  # image_feat of all images,when are fed into net ,must divide
        spt_feat[i, :, :] = get_spt_feat(feat_file_path)  # do not know their id ,so cannot related to the query,so,format_train.pkl->iid->FEAT_DIR


        i += 1
    '''
    return img_feat, spt_feat, p, gt_boxes

class DDPN_NET(nn.Module):
    def __init__(self):
        super(DDPN_NET, self).__init__()

        

    def forward(self, img_feat, spt_feat, p):  # img_feat:batchsize* 100*2048;spt_feat:batchsize* 100*5
        img_feat = F.normalize(img_feat, p=2, dim=2)
        v = self.get_feat_v(img_feat, spt_feat)  # v:batchsize* 100*(2048+5)




    def get_feat_v(self, img_feat, spt_feat):
        feat_v = torch.cat((img_feat, spt_feat), 2)
        return feat_v


if __name__ == '__main__':

    '''anno_path = ANNO_DIR+'format_train.pkl'
    anno = pickle.load(open(anno_path, 'rb'), encoding='iso-8859-1')
    print(anno.keys())

    anno_list = []
    for name in anno.keys():
        anno_list.append(anno[name])
    print(anno_list.__len__())

    iid = anno_list[0]['iid']
    print(iid)
    file_path = FEAT_DIR+COCO_PREFIX+str(iid)+FEAT_SUFFIX


    file = '/data/chenchao/data/mscoco/features/bottom-up-feats/train2014/COCO_train2014_000000210642.jpg.npz'

    spt_feat = get_spt_feat(file)

    print(spt_feat.shape)
    print(spt_feat[0, :])

    img_feat = get_img_feat(file)

    print(img_feat[:, 0])
    '''

    '''
    str'dark horse in center looking at camera'
    '''

    # format_path = '/home/lipin/code/DDPN/refcoco/format_train.pkl'
    # words = get_p(format_path)
    #
    # words_list = str2list(words)

    '''
    idx2token
    len:9368;
    dict{0:'<blank>',1:'<unk>',2:'<s>',3:'</s>',4:'the',5:'lady',6:'with',7:'blue',8:'shirt',9:'back',10:'to',....}
    '''
    idx2token = pickle.load(open('/home/lipin/code/DDPN/refcoco/query_dict/idx2token.pkl', 'rb'), encoding='iso-8859-1')

    '''
    token2idx
    len:9368
    dict{'raining':1048,'yellos':8494,'yellow':192,...}
    '''
    token2idx_dict = pickle.load(open('/home/lipin/code/DDPN/refcoco/query_dict/token2idx.pkl', 'rb'), encoding='iso-8859-1')

    '''
    special_words
    len:4
    list:['blank','unk','<s>','</s>']
    '''
    special_words = pickle.load(open('/home/lipin/code/DDPN/refcoco/query_dict/special_words.pkl', 'rb'), encoding='iso-8859-1')

    '''
    word_freq
    len:9368
    dict{0:1,1:1,2:1,3:1,4:15958,5:2167,6:6170,....}
    '''
    word_freq = pickle.load(open('/home/lipin/code/DDPN/refcoco/query_dict/word_freq.pkl', 'rb'), encoding='iso-8859-1')

    format_train_data_dict = pickle.load(open('/home/lipin/code/DDPN/refcoco/format_train.pkl', 'rb'), encoding='iso-8859-1')

    format_anno_list = []
    for name in format_train_data_dict.keys():
        format_anno_list.append(format_train_data_dict[name])

    len1 = format_anno_list.__len__()

    print('len1', len1)

    iter = int(len1/BATCHSIZE)  # dozens of data abandoned
    ignored_num = len1 - iter*BATCHSIZE
    print('ignored_num:', ignored_num)

    for j in range(iter-1):
        print('iteration:', j)
        img_feat, spt_feat, p, gt_boxes = net_input(format_anno_list[j*BATCHSIZE:(j+1)*BATCHSIZE-1], token2idx_dict)


    print()


