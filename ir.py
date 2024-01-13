import os

import torch
from collections import defaultdict
import PIL.Image as Image
from torchvision.transforms import transforms
from typing import List

f = open('./celebA_ir/celebA_anno_query.csv', 'r')
query_lines = f.readlines()[1:]
f.close()
query_lines = [x.strip().split(',') for x in query_lines]
query_img_names = [x[0] for x in query_lines]

query_dict = defaultdict(list)

for img_name, img_class in query_lines:
    query_dict[img_class].append(img_name)


distractors_img_names = os.listdir('./celebA_ir/celebA_distractors')


def compute_embeddings(model: torch.nn.Module, images_list: List[Image]) -> torch.Tensor:
    """
    compute embeddings from the trained model for list of images.
    params:
        model: trained nn model that takes images and outputs embeddings
        images_list: list of images paths to compute embeddings for
    output:
        list: list of model embeddings. Each embedding corresponds to images
          names from images_list
    """
    images_to_batch_tensor = []
    tf = transforms.Compose([transforms.ToTensor()])
    for image in images_list:
        transformed_image = tf(image).unsqueeze(0)
        images_to_batch_tensor.append(transformed_image)
    image_batch = torch.cat(images_to_batch_tensor, dim=0)
    return model(image_batch)


def compute_cosine_query_pos(query_dict: list, query_img_names, query_embeddings: torch.Tensor):
    """
    compute cosine similarity between positive pairs from query (stage 1)
    params:
        query_dict: dict {class: [image_name_1, image_name_2, ...]}. Key: class in
                the dataset. Value: images corresponding to that class
        query_img_names: list of images names
        query_embeddings: list of embeddings corresponding to query_img_names
    output:
        list of floats: similarities between embeddings corresponding
                    to the same people from query list
    """
    similarities = []
    for i in range(person_photos_len):
        for j in range(i, person_photos_len):
            if i != j:
                tensor_i = embeddings[i].detach()
                tensor_j = embeddings[j].detach()
                cosine_similarity = torch.nn.CosineSimilarity(dim=0)
                cos = cosine_similarity(tensor_i, tensor_j)
                similarities.append(cos.item())
    return similarities


def compute_cosine_query_neg(query_dict, query_img_names, query_embeddings: torch.Tensor):
    """
    compute cosine similarity between negative pairs from query and distractors
    :param query_dict:
    :param query_img_names:
    :param query_embeddings:
    :return:
    """
    pass


def compute_cosine_query_distractors(query_embeddings: torch.Tensor, distractors_embeddings: torch.Tensor):
    """
    compute cosine similarities between negative pairs from query and distractors
    :param query_embeddings:
    :param distractors_embeddings:
    :return:
    """
    pass


def compute_ir(cosine_query_pos, cosine_query_neg, cosine_query_distractors, fpr=0.1):
    pass
