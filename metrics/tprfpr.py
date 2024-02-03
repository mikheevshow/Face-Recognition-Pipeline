import torch
from torch import Tensor
from torch.nn import Module, CosineSimilarity
from torch.utils.data import DataLoader

from celeb_query_dataset import CelebQueryDataset
from celeb_distractors_dataset import CelebDistractorsDataset


def __compute_embeddings(model: Module, dataloader: DataLoader, device: torch.device) -> Tensor:
    """
    Compute embeddings from the trained model for images from celebA_ir dataset
    :param model: trained model
    :param dataloader: query or distractors dataloader
    :param device:
    :return:
    """
    model.eval()
    embeddings = None
    with torch.inference_mode():
        for _, images_batch in enumerate(dataloader):
            if isinstance(images_batch, list):
                images_batch_tensor = images_batch[0]
            else:
                images_batch_tensor = images_batch
            model = model.to(device)
            images_batch_tensor = images_batch_tensor.to(device)
            batch_of_embeddings = model.get_embedding(images_batch_tensor)
            if embeddings is None:
                embeddings = batch_of_embeddings
            else:
                embeddings = torch.cat((embeddings, batch_of_embeddings), dim=0)
    return embeddings.detach()


def __compute_cosine_query_pos(query_dataset: CelebQueryDataset, query_embeddings: Tensor) -> list:
    """
    Compute cosine similarities between positive pairs from query (stage 1)
    :param query_dataset:
    :param query_embeddings:
    :return:
    """
    cosine_similarity = CosineSimilarity(dim=0)
    query_person_ids = query_dataset.get_person_ids()
    pos_similarities = []
    for person_id in query_person_ids:
        query_embedding_indices = query_dataset.get_image_indices(person_id)
        query_embedding_indices_len = len(query_embedding_indices)
        person_query_embeddings = query_embeddings[min(query_embedding_indices): max(query_embedding_indices) + 1]

        for i in range(query_embedding_indices_len):
            for j in range(query_embedding_indices_len):
                if i < j:
                    i_tensor = person_query_embeddings[i]
                    j_tensor = person_query_embeddings[j]
                    sim = cosine_similarity(i_tensor, j_tensor).item()
                    pos_similarities.append(sim)
    return pos_similarities


def __compute_cosine_query_neg(query_dataset: CelebQueryDataset, query_embeddings: Tensor) -> list:
    """
    Compute cosine similarities between negative pairs from query (stage 2)
    :param query_dataset:
    :param query_embeddings:
    :return:
    """
    cosine_similarity = CosineSimilarity(dim=0)
    query_person_ids = query_dataset.get_person_ids()
    neg_similarities = []
    for person_id_i in query_person_ids:
        for person_id_j in query_person_ids:
            if person_id_i < person_id_j:
                person_id_i_embeddings_indices = query_dataset.get_image_indices(person_id_i)
                person_id_i_len = len(person_id_i_embeddings_indices)
                person_i_from_index = min(person_id_i_embeddings_indices)
                person_i_to_index = max(person_id_i_embeddings_indices) + 1
                person_id_i_embeddings = query_embeddings[person_i_from_index:person_i_to_index]

                person_id_j_embeddings_indices = query_dataset.get_image_indices(person_id_j)
                person_id_j_len = len(person_id_j_embeddings_indices)
                person_j_from_index = min(person_id_j_embeddings_indices)
                person_j_to_index = max(person_id_j_embeddings_indices) + 1
                person_id_j_embeddings = query_embeddings[person_j_from_index:person_j_to_index]

                for i in range(person_id_i_len):
                    for j in range(person_id_j_len):
                        i_tensor = person_id_i_embeddings[i]
                        j_tensor = person_id_j_embeddings[j]
                        sim = cosine_similarity(i_tensor, j_tensor).item()
                        neg_similarities.append(sim)
    return neg_similarities


def __compute_cosine_query_distractors(query_dataset: CelebQueryDataset,
                                       distractors_dataset: CelebDistractorsDataset,
                                       query_embeddings: Tensor,
                                       distractor_embeddings: Tensor) -> list:
    """
    Compute cosine similarities between negative pairs from query and distractors
    :param query_dataset:
    :param distractors_dataset:
    :param query_embeddings:
    :param distractor_embeddings:
    :return:
    """
    cosine_similarity = CosineSimilarity(dim=0)
    query_distractor_similarities = []
    for i in range(len(query_dataset)):
        for j in range(len(distractors_dataset)):
            i_tensor = query_embeddings[i]
            j_tensor = distractor_embeddings[j]
            sim = cosine_similarity(i_tensor, j_tensor).item()
            query_distractor_similarities.append(sim)
    return query_distractor_similarities


def compute_ir_metric(model: Module, fpr=0.1, device=torch.device("cuda")):
    """
    Compute the identification rate metric. Import this function into your code.
    :param model:
    :param fpr:
    :param device:
    :return:
    """
    model.eval()
    model.to(device)

    query_dataset = CelebQueryDataset()
    distractors_dataset = CelebDistractorsDataset()

    query_dataloader = DataLoader(dataset=query_dataset, batch_size=100)
    distractors_dataloader = DataLoader(dataset=distractors_dataset, batch_size=100)

    query_embeddings = __compute_embeddings(model, query_dataloader, device)
    distractor_embeddings = __compute_embeddings(model, distractors_dataloader, device)

    pos_similarities = __compute_cosine_query_pos(query_dataset, query_embeddings)
    neg_similarities = __compute_cosine_query_neg(query_dataset, query_embeddings)
    query_distractor_similarities = __compute_cosine_query_distractors(query_dataset, distractors_dataset, query_embeddings, distractor_embeddings)

    false_pairs = neg_similarities + query_distractor_similarities
    false_pairs = sorted(false_pairs)

    # Acceptable amount of false pairs
    N = int(fpr * len(false_pairs))

    threshold_similarity = false_pairs[N]

    metric_value: int = 0
    for s in pos_similarities:
        if s < threshold_similarity:
            metric_value += 1

    return threshold_similarity, metric_value

