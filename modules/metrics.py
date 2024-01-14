import torch
import numpy as np

def euclidean_distance(vector_a, vector_b):
    """
    Calculate euclidean distance for 1d vectors
    """
    if type(vector_a) == torch.Tensor:
        vector_a = vector_a.detach().numpy()
    if type(vector_b) == torch.Tensor:
        vector_b = vector_b.detach().numpy()
    
    assert len(vector_a.shape) == 1
    assert len(vector_b.shape) == 1
    assert vector_a.shape[0] == vector_b.shape[0]
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a, vector_b):
    """
    Calculate manhattan distance for 1d vectors
    """
    if type(vector_a) == torch.Tensor:
        vector_a = vector_a.detach().numpy()
    if type(vector_b) == torch.Tensor:
        vector_b = vector_b.detach().numpy()
    
    assert len(vector_a.shape) == 1
    assert len(vector_b.shape) == 1
    assert vector_a.shape[0] == vector_b.shape[0]
    return np.sum(np.abs(vector_a - vector_b))

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity for 1d vectors
    """
    if type(vector_a) == torch.Tensor:
        vector_a = vector_a.detach().numpy()
    if type(vector_b) == torch.Tensor:
        vector_b = vector_b.detach().numpy()
    
    assert len(vector_a.shape) == 1
    assert len(vector_b.shape) == 1
    assert vector_a.shape[0] == vector_b.shape[0]
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


class ScoreModule:
    def __init__(self, approach):
        self.approach = approach
        if self.approach in ['vl_retrieval', 'text_retrieval', 'image_retrieval']:
            self.prediction_list = {i: 0 for i in range(10)}
        elif self.approach == 'qa_retrieval':
            self.correct_cnt = 0
        self.total_cnt = 0
    
    def add(self, golden_image_index, predictions):
        if predictions is None:
            return

        if self.approach in ['vl_retrieval', 'text_retrieval', 'image_retrieval']:
            pred = np.where(np.array(predictions)==golden_image_index)[0][0]
            self.prediction_list[pred] += 1
        elif self.approach == 'qa_retrieval':
            if golden_image_index == int(predictions):
                self.correct_cnt += 1
        
        self.total_cnt += 1
    
    def accuracy_score(self):
        if self.approach in ['vl_retrieval', 'text_retrieval', 'image_retrieval']:
            return self.prediction_list[0] / self.total_cnt
        elif self.approach == 'qa_retrieval':
            return self.correct_cnt / self.total_cnt

    def mrr_score(self):
        if self.approach in ['vl_retrieval', 'text_retrieval', 'image_retrieval']:
            mrr = np.sum([self.prediction_list[i] * (1/(i+1)) for i in range(10)])
            return mrr / self.total_cnt
        elif self.approach == 'qa_retrieval':
            raise ValueError("MRR Score is not supported for QA Retrieval")