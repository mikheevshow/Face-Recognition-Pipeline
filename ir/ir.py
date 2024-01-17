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

    person_similarities = []
    for person_id in query_dataset.get_person_ids():
        query_embedding_indices = query_dataset.get_image_indices(person_id)
        query_embedding_indices_len = len(query_embedding_indices)
        person_query_embeddings = query_embeddings[min(query_embedding_indices) : max(query_embedding_indices)]

        for i in range(query_embedding_indices_len):
            for j in range(i, query_embedding_indices_len):
                if i != j:
                    i_tensor = person_query_embeddings[i]
                    j_tensor = person_query_embeddings[j]
                    cosine_similarity = cosine_similarity(i_tensor, j_tensor).item()
                    person_similarities.append(cosine_similarity)

    # calculate others
    return 0.0

