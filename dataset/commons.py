available_modes = ['train', 'val', 'test']
availaible_file_locations = ['Kaggle', 'Local']

kaggle_root_data_path = '/kaggle/input/celeba-train-500/celebA_train_500/'
kaggle_images_path = '/kaggle/input/celeba-train-500/celebA_train_500/celebA_imgs/'


local_root_data_path = '../celebA_train_500/'
local_images_path = '../celebA_train_500/celebA_imgs/'


def get_image_path(env: str = 'Kaggle') -> str:
    return local_root_data_path

