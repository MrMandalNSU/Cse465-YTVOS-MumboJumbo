import os
import router

'''
json_path = '/home/shashank/shashank/datasets/YoutubeVos/train-004/train/meta.json'
JPEGtrain_5fps = '/home/shashank/shashank/datasets/YoutubeVos/train-004/train/JPEGImages/'
Anntrain_5fps = '/home/shashank/shashank/datasets/YoutubeVos/train-004/train/Annotations/'

modelPth = '/home/shashank/shashank/AdvCV/Assign2/weights/youtubeVOSModel_trial280.pth'

JPEGValidation = '/home/shashank/shashank/datasets/YoutubeVos/YouTubeVOS_2018-20190401T153058Z-001/YouTubeVOS_2018/valid/JPEGImages/'
AnnValidation = '/home/shashank/shashank/datasets/YoutubeVos/YouTubeVOS_2018-20190401T153058Z-001/YouTubeVOS_2018/valid/Annotations/'
validation_json = '/home/shashank/shashank/datasets/YoutubeVos/YouTubeVOS_2018-20190401T153058Z-001/YouTubeVOS_2018/valid/meta.json'

indivAnnotation = '/home/shashank/shashank/AdvCV/YoutubeVOS_submission/'
indivAnnotation_check = '/home/shashank/shashank/AdvCV/YoutubeVOS_merged_thrsh04/'
thresh = 0.4
'''

# block added by Team-MumboJumbo
cuda_enable = True   # CPU: False, GPU: True
shanto_debug = True   # enabling custom changes
valid_subset = -1  # instead of trying all validation folder, try only 5. To try all folder put -1
epoch = 50


json_path = os.path.join(router.tsub, 'meta.json')
JPEGtrain_5fps = os.path.join(router.tsub, 'JPEGImages/')
Anntrain_5fps = os.path.join(router.tsub, 'Annotations/')

modelPth = os.path.join(router.model_root, 'youtubeVOSModel_trial_3_1.pth')

JPEGValidation = os.path.join(router.valid, 'JPEGImages/')
AnnValidation = os.path.join(router.valid, 'Annotations/')
validation_json = os.path.join(router.valid, 'meta.json')

indivAnnotation = router.YoutubeVOS_submission
indivAnnotation_check = router.YoutubeVOS_merged_thrsh04
thresh = 0.4
