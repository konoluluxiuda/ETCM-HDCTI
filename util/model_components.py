import numpy as np


def build_regularization_loss(
        tf_module,
        weights,
        compound_embeddings,
        protein_embeddings,
        weight_decay,
        compound_regularization,
        protein_regularization):
    terms = [
        float(weight_decay) * tf_module.nn.l2_loss(weight)
        for weight in weights.values()
    ]
    terms.extend([
        float(compound_regularization) * tf_module.nn.l2_loss(compound_embeddings),
        float(protein_regularization) * tf_module.nn.l2_loss(protein_embeddings),
    ])
    return tf_module.add_n(terms)


def context_interaction_scores(
        compound_embeddings,
        protein_embeddings,
        compound_contexts,
        protein_contexts,
        compound_disease_weight,
        herb_protein_weight,
        herb_disease_weight):
    compound_embeddings = np.asarray(compound_embeddings)
    protein_embeddings = np.asarray(protein_embeddings)
    compound_contexts = np.asarray(compound_contexts)
    protein_contexts = np.asarray(protein_contexts)

    scores = compound_embeddings.dot(protein_embeddings.transpose())
    scores += (compound_embeddings * compound_disease_weight).dot(protein_contexts.transpose())
    scores += (compound_contexts * herb_protein_weight).dot(protein_embeddings.transpose())
    scores += (compound_contexts * herb_disease_weight).dot(protein_contexts.transpose())
    return scores
