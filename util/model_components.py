import numpy as np


CONTEXT_TERM_CONFIG = {
    'compound_disease': 'context.compound_disease',
    'herb_protein': 'context.herb_protein',
    'herb_disease': 'context.herb_disease',
}


def _config_bool(config, key, default):
    if not config.contains(key):
        return default
    value = str(config[key]).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError('Invalid boolean value for %s: %s' % (key, config[key]))


def resolve_context_terms(config):
    master_enabled = _config_bool(config, 'context.interaction', False)
    return {
        term: master_enabled and _config_bool(config, key, True)
        for term, key in CONTEXT_TERM_CONFIG.items()
    }


def resolve_early_stopping(config):
    enabled = _config_bool(config, 'early.stopping', False)
    settings = {
        'enabled': enabled,
        'ratio': float(config['validation.ratio']) if config.contains('validation.ratio') else 0.1,
        'metric': str(config['validation.metric']).strip().lower()
        if config.contains('validation.metric') else 'aupr',
        'interval': int(config['validation.interval'])
        if config.contains('validation.interval') else 2,
        'patience': int(config['validation.patience'])
        if config.contains('validation.patience') else 5,
        'min_delta': float(config['validation.min.delta'])
        if config.contains('validation.min.delta') else 0.0001,
    }
    if not 0.0 < settings['ratio'] < 1.0:
        raise ValueError('validation.ratio must be between 0 and 1.')
    if settings['metric'] not in {'aupr', 'auc'}:
        raise ValueError('validation.metric must be AUPR or AUC.')
    if settings['interval'] <= 0:
        raise ValueError('validation.interval must be positive.')
    if settings['patience'] <= 0:
        raise ValueError('validation.patience must be positive.')
    if settings['min_delta'] < 0:
        raise ValueError('validation.min.delta cannot be negative.')
    return settings


def resolve_pair_decoder(config):
    decoder_type = (
        str(config['pair.decoder']).strip().lower()
        if config.contains('pair.decoder') else 'dot'
    )
    if decoder_type not in {'dot', 'bilinear', 'mlp'}:
        raise ValueError('pair.decoder must be dot, bilinear, or mlp.')
    hidden_size = (
        int(config['pair.mlp.hidden'])
        if config.contains('pair.mlp.hidden') else 64
    )
    prediction_batch_size = (
        int(config['pair.prediction.batch.size'])
        if config.contains('pair.prediction.batch.size') else 65536
    )
    if hidden_size <= 0:
        raise ValueError('pair.mlp.hidden must be positive.')
    if prediction_batch_size <= 0:
        raise ValueError('pair.prediction.batch.size must be positive.')
    return {
        'type': decoder_type,
        'hidden_size': hidden_size,
        'prediction_batch_size': prediction_batch_size,
    }


class EarlyStoppingTracker(object):
    def __init__(self, patience, min_delta):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value = None
        self.best_epoch = None
        self.stale_checks = 0

    def update(self, value, epoch):
        value = float(value)
        improved = self.best_value is None or value > self.best_value + self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = int(epoch)
            self.stale_checks = 0
        else:
            self.stale_checks += 1
        return improved, self.stale_checks >= self.patience


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
        herb_disease_weight,
        enabled_terms=None):
    compound_embeddings = np.asarray(compound_embeddings)
    protein_embeddings = np.asarray(protein_embeddings)
    compound_contexts = np.asarray(compound_contexts)
    protein_contexts = np.asarray(protein_contexts)

    if enabled_terms is None:
        enabled_terms = {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        }
    scores = compound_embeddings.dot(protein_embeddings.transpose())
    if enabled_terms.get('compound_disease', False):
        scores += (compound_embeddings * compound_disease_weight).dot(protein_contexts.transpose())
    if enabled_terms.get('herb_protein', False):
        scores += (compound_contexts * herb_protein_weight).dot(protein_embeddings.transpose())
    if enabled_terms.get('herb_disease', False):
        scores += (compound_contexts * herb_disease_weight).dot(protein_contexts.transpose())
    return scores


def context_interaction_pair_scores(
        compound_embeddings,
        protein_embeddings,
        compound_contexts,
        protein_contexts,
        compound_indices,
        protein_indices,
        compound_disease_weight,
        herb_protein_weight,
        herb_disease_weight,
        enabled_terms=None,
        decoder_type='dot',
        decoder_weights=None):
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')

    compounds = np.asarray(compound_embeddings)[compound_indices]
    proteins = np.asarray(protein_embeddings)[protein_indices]
    herb_contexts = np.asarray(compound_contexts)[compound_indices]
    disease_contexts = np.asarray(protein_contexts)[protein_indices]
    if enabled_terms is None:
        enabled_terms = {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        }

    scores = pair_decoder_scores(
        compounds, proteins, decoder_type=decoder_type, decoder_weights=decoder_weights
    )
    if enabled_terms.get('compound_disease', False):
        scores += np.sum(compounds * disease_contexts * compound_disease_weight, axis=1)
    if enabled_terms.get('herb_protein', False):
        scores += np.sum(herb_contexts * proteins * herb_protein_weight, axis=1)
    if enabled_terms.get('herb_disease', False):
        scores += np.sum(herb_contexts * disease_contexts * herb_disease_weight, axis=1)
    return scores


def pair_decoder_scores(
        compound_vectors,
        protein_vectors,
        decoder_type='dot',
        decoder_weights=None):
    compounds = np.asarray(compound_vectors)
    proteins = np.asarray(protein_vectors)
    if compounds.shape != proteins.shape:
        raise ValueError('Compound and protein vectors must have the same shape.')
    decoder_type = str(decoder_type).strip().lower()
    decoder_weights = decoder_weights or {}

    dot_scores = np.sum(compounds * proteins, axis=1)
    if decoder_type == 'dot':
        return dot_scores
    if decoder_type == 'bilinear':
        if 'decoder_bilinear' not in decoder_weights:
            raise ValueError('Bilinear decoder weight is missing.')
        return np.sum(
            compounds.dot(np.asarray(decoder_weights['decoder_bilinear'])) * proteins,
            axis=1,
        )
    if decoder_type == 'mlp':
        required = (
            'decoder_mlp_hidden', 'decoder_mlp_hidden_bias',
            'decoder_mlp_output', 'decoder_mlp_output_bias',
        )
        missing = [key for key in required if key not in decoder_weights]
        if missing:
            raise ValueError('MLP decoder weights are missing: %s.' % ', '.join(missing))
        features = np.concatenate(
            [compounds, proteins, compounds * proteins, np.abs(compounds - proteins)],
            axis=1,
        )
        hidden = features.dot(np.asarray(decoder_weights['decoder_mlp_hidden']))
        hidden += np.asarray(decoder_weights['decoder_mlp_hidden_bias'])
        hidden = np.where(hidden >= 0, hidden, 0.2 * hidden)
        residual = hidden.dot(np.asarray(decoder_weights['decoder_mlp_output']))
        residual += np.asarray(decoder_weights['decoder_mlp_output_bias'])
        return dot_scores + np.reshape(residual, (-1,))
    raise ValueError('Unsupported pair decoder: %s.' % decoder_type)
