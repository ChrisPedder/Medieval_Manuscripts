"""
Define class to do train/validation/test split of different datasets
"""

### Do train/test split of dataset for training purposes, retain specific
### split used for future use/investigation

def generate_train_test_split_files(train_test_split, set_size):
    # Set size of train-test split
    split = round((1-train_test_split) * set_size)

    # get list of files in the raw data directory
    root = Path.cwd()
    MS_folder = str(root.parent.parent) + '/' + 'Projects/Medieval_Manuscripts/data/interim/MS157' + '/'
    CLaMM_folder = str(root.parent.parent) + '/' + 'Projects/Medieval_Manuscripts/data/interim/CLaMM' + '/'

    # take random sample of size SET_SIZE from examples
    MS_sample_list = random.sample(glob.glob(MS_folder + '/*'), set_size)
    CLaMM_sample_list = random.sample(glob.glob(CLaMM_folder + '/*'),
                                     set_size)

    MS_tr_files = []
    CLaMM_tr_files = []
    for i in range(split):
        MS_tr_files.append(MS_sample_list[i])
        CLaMM_tr_files.append(CLaMM_sample_list[i])

    MS_te_files = []
    CLaMM_te_files = []
    for i in range(split,set_size):
        MS_te_files.append(MS_sample_list[i])
        CLaMM_te_files.append(CLaMM_sample_list[i])

    return [MS_tr_files, MS_te_files, CLaMM_tr_files, CLaMM_te_files]

def add_path(string):
    """
    Helper function for following routine to generate list of train/test
    target directories required by Keras...
    """
    root = Path.cwd()
    top_path = str(root.parent.parent)
    return top_path + '/' + string + '/'

def generate_target_train_test_directories(set_size):
    """
    Create set of routines for making a list of the target directories for
    copied train/test split files
    """

    target_list = ['data/processed/set_size_' + str(set_size),\
                   'data/processed/set_size_' + str(set_size) + '/train',\
                   'data/processed/set_size_' + str(set_size) + '/test',\
                   'data/processed/set_size_' + str(set_size) + '/train/' + 'MS157',\
                   'data/processed/set_size_' + str(set_size) + '/test/' + 'MS157',\
                   'data/processed/set_size_' + str(set_size) + '/train/' + 'CLaMM',\
                   'data/processed/set_size_' + str(set_size) + '/test/' + 'CLaMM']

    train_test_directories = []
    for extension in target_list:
        train_test_directories.append(add_path(extension))

    return train_test_directories

def do_train_test_split(split, set_size):
    """
    Do train-test split of data into subfolders required for Keras
    retraining format.
    """
    import ipdb; ipdb.set_trace()
    file_locations = generate_train_test_split_files(split, set_size)

    # generate list of target directories to copy files to
    path_list = generate_target_train_test_directories(set_size)

    if all(os.path.exists(x) for x in path_list):
        print('Train test split already exists for this set size. Using existing split...')
        return

    else:
        #check chosen directory exists, if not create it
        for path in path_list:
            os.mkdir(path)

        # copy files to train and test directories
        for i in range(4):
            for filename in file_locations[i]:
                shutil.copy2(filename, path_list[i+3])
                print("File named {} copied to directory {}".format(filename,
                      file_locations[i]))

        return
