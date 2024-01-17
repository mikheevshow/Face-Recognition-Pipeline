import torch
from torch import Tensor
from torch.nn import Module, CosineSimilarity
from torch.utils.data import DataLoader

from celeb_query_dataset import CelebQueryDataset
from celeb_distractors_dataset import CelebDistractorsDataset


def compute_embeddings(model: Module, dataloader: DataLoader, device: torch.device) -> Tensor:
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
            batch_of_embeddings = model(images_batch_tensor)
            if embeddings is None:
                embeddings = batch_of_embeddings
            else:
                embeddings = torch.cat((embeddings, batch_of_embeddings), dim=0)
    return embeddings.detach()


def compute_ir(model: Module, fpr=0.1, device=None):
    if device is None:
        device = torch.device("cpu")

    query_dataset = CelebQueryDataset()
    distractors_dataset = CelebDistractorsDataset()

    query_dataloader = DataLoader(dataset=query_dataset, batch_size=100)
    distractors_dataloader = DataLoader(dataset=distractors_dataset, batch_size=100)

    query_embeddings = compute_embeddings(model, query_dataloader, device)
    distractor_embeddings = compute_embeddings(model, distractors_dataloader, device)

    # calculate same cosine similarities
    cosine_similarity = CosineSimilarity(dim=0)

    query_person_ids = query_dataset.get_person_ids()

    # Same query person similarity
    person_similarities = []
    for person_id in query_person_ids:
        query_embedding_indices = query_dataset.get_image_indices(person_id)
        query_embedding_indices_len = len(query_embedding_indices)
        person_query_embeddings = query_embeddings[min(query_embedding_indices): max(query_embedding_indices) + 1]

        for i in range(query_embedding_indices_len):
            for j in range(i, query_embedding_indices_len):
                if i != j:
                    i_tensor = person_query_embeddings[i]
                    j_tensor = person_query_embeddings[j]
                    cosine_similarity = cosine_similarity(i_tensor, j_tensor).item()
                    person_similarities.append(cosine_similarity)

    # Different query persons similarities
    cross_query_similarities = []
    for person_id_i in query_person_ids:
        for person_id_j in query_person_ids:
            if person_id_i != person_id_j:
                person_id_i_embeddings_indices = query_dataset.get_image_indices(person_id_i)
                person_id_i_len = len(person_id_i_embeddings_indices)
                person_i_from_index = min(person_id_i_embeddings_indices)
                person_i_to_index = max(person_id_i_embeddings_indices) + 1
                person_id_i_embeddings = query_embeddings[person_i_from_index:person_i_to_index]

                person_id_j_embeddings_indices = query_dataset.get_image_indices(person_id_j)
                person_id_j_len = len(person_id_i_embeddings_indices)
                person_j_from_index = min(person_id_j_embeddings_indices)
                person_j_to_index = max(person_id_j_embeddings_indices) + 1
                person_id_j_embeddings = query_embeddings[person_j_from_index:person_j_to_index]

                for i in range(person_id_i_len):
                    for j in range(person_id_j_len):
                        i_tensor = person_id_i_embeddings[i]
                        j_tensor = person_id_j_embeddings[j]
                        cosine_similarity = cosine_similarity(i_tensor, j_tensor).item()
                        cross_query_similarities.append(cosine_similarity)

    # Similarity between query and distractor persons
    query_distractor_similarities = []
    for i in range(len(query_dataset)):
        for j in range(len(distractors_dataset)):
            i_tensor = query_embeddings[i]
            j_tensor = distractor_embeddings[j]
            cosine_similarity = cosine_similarity(i_tensor, j_tensor).item()
            query_distractor_similarities.append(cosine_similarity)

    false_pairs = cross_query_similarities + query_distractor_similarities
    false_pairs = sorted(false_pairs)

    # Acceptable amount of false pairs
    N = int(fpr * len(false_pairs))

    threshold_similarity = false_pairs[N]

    metric_value: int = 0
    for s in person_similarities:
        if s < threshold_similarity:
            metric_value += 1

    return metric_value

