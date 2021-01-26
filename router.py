import os

PROJECT_NAME = 'video_object_segmentation'


def mkdir_p(dir, verbose=False, backup_existing=False, if_contains=None):
    '''make a directory (dir) if it doesn't exist'''
    # todo: add recursive directory creation
    # todo: give a msg when dir exist but empty
    # dir: given full path only create last name as directory
    # if_contains: backup_existing True if_contains is a list. If there is any file/directory in the
    #              first (not recursive) pass match with if_contains then backup
    if not os.path.exists(dir):  # directory does not exist
        if verbose is True: errprint(f'Created new dir named: {dir}')
        os.mkdir(dir)
    else:  # dir exist
        need_backup = False
        # check if backup is needed
        if backup_existing and len(os.listdir(dir)) > 0:
            # check contains helper
            def helper_contains():
                # check if if_contains is a list
                if not isinstance(if_contains, list):
                    raise ValueError('if_contains in mkdir_p is not a list')

                files_inside = os.listdir(dir)
                for pat in if_contains:
                    if any(pat in file for file in files_inside):
                        return True
                return False

            if (if_contains is None) or (if_contains and helper_contains()):
                need_backup = True

        if need_backup:
            # find new path that doesn't exist
            for i in range(10000):
                new_dir_path = f'{dir}_{i}'
                if not os.path.exists(new_dir_path):
                    break
            # renaming directory
            if verbose:
                errprint(f'Moving dir {dir} -> {new_dir_path}')
            os.rename(src=dir, dst=new_dir_path)
            # now creating dir
            os.mkdir(dir)
    return dir


def fix_root(project_name=PROJECT_NAME):
    root = os.getcwd()
    first, last = os.path.split(root)
    if last == project_name:
        return root
    else:
        return first

# project_root = os.getcwd()
project_root = fix_root()

data_root = os.path.join(project_root, 'data') # all data
model_root = os.path.join(data_root, 'model')  # ml model
fig_root = os.path.join(data_root, 'fig')  # figure
bigdata_root = os.path.join(data_root, 'big')  # big size data
expdata_root = os.path.join(data_root, 'exp')  # experimental data
input_root = os.path.join(data_root, 'input')  # input to program. e.g. programs, io examples
tmp_root = os.path.join(data_root, 'tmp')

job_dir = os.path.join(project_root, '.job')  # jobs

train = os.path.join(project_root, 'raw_data/train')
tsub = os.path.join(project_root, 'raw_data/train-subset')

valid = os.path.join(project_root, 'raw_data/valid')
YoutubeVOS_submission = os.path.join(data_root, '')
YoutubeVOS_merged_thrsh04 = os.path.join(data_root, 'YoutubeVOS_merged_thrsh04')


# creating all directory
for folder in [data_root, model_root, fig_root, bigdata_root, expdata_root, input_root, job_dir, tmp_root,
               YoutubeVOS_submission]:
    mkdir_p(folder)

