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


def resolve_herb_context_attention(config):
    mode = (
        str(config['context.herb_protein.mode']).strip().lower()
        if config.contains('context.herb_protein.mode') else 'static'
    )
    valid_modes = {'static', 'target_attention', 'target_residual_attention'}
    if mode not in valid_modes:
        raise ValueError(
            'context.herb_protein.mode must be static, target_attention, '
            'or target_residual_attention.'
        )
    temperature = (
        float(config['context.herb_attention.temperature'])
        if config.contains('context.herb_attention.temperature') else 1.0
    )
    if temperature <= 0:
        raise ValueError('context.herb_attention.temperature must be positive.')
    return {
        'mode': mode,
        'temperature': temperature,
    }


def resolve_counterfactual_context(config):
    enabled = _config_bool(config, 'counterfactual.context', False)
    settings = {
        'enabled': enabled,
        'weight': (
            float(config['counterfactual.weight'])
            if config.contains('counterfactual.weight') else 0.05
        ),
        'margin': (
            float(config['counterfactual.margin'])
            if config.contains('counterfactual.margin') else 0.2
        ),
        'draws': (
            int(config['counterfactual.draws'])
            if config.contains('counterfactual.draws') else 20
        ),
        'seed': (
            int(config['counterfactual.seed'])
            if config.contains('counterfactual.seed') else 42026
        ),
        'match': (
            str(config['counterfactual.match']).strip().lower()
            if config.contains('counterfactual.match')
            else 'exact_hc_degree_disjoint'
        ),
    }
    if settings['weight'] < 0:
        raise ValueError('counterfactual.weight cannot be negative.')
    if settings['margin'] <= 0:
        raise ValueError('counterfactual.margin must be positive.')
    if settings['draws'] <= 0:
        raise ValueError('counterfactual.draws must be positive.')
    if settings['match'] != 'exact_hc_degree_disjoint':
        raise ValueError(
            'counterfactual.match must be exact_hc_degree_disjoint.'
        )
    if enabled and settings['weight'] == 0:
        raise ValueError(
            'counterfactual.weight must be positive when counterfactual.context=True.'
        )
    return settings


def resolve_context_mask_training(config):
    enabled = _config_bool(config, 'context.mask.training', False)
    side = (
        str(config['context.mask.side']).strip().lower()
        if config.contains('context.mask.side') else 'compound'
    )
    weight = (
        float(config['context.mask.weight'])
        if config.contains('context.mask.weight') else 0.1
    )
    if side not in {'compound', 'protein', 'both'}:
        raise ValueError('context.mask.side must be compound, protein, or both.')
    if weight < 0:
        raise ValueError('context.mask.weight cannot be negative.')
    if enabled and weight == 0:
        raise ValueError(
            'context.mask.weight must be positive when context.mask.training=True.'
        )
    return {
        'enabled': enabled,
        'side': side,
        'weight': weight,
    }


def resolve_support_router(config):
    enabled = _config_bool(config, 'support.router', False)
    mode = (
        str(config['support.router.mode']).strip().lower()
        if config.contains('support.router.mode') else 'monotonic_residual'
    )
    pseudo_cold_ratio = (
        float(config['support.router.pseudo.cold.ratio'])
        if config.contains('support.router.pseudo.cold.ratio') else 0.1
    )
    seed = (
        int(config['support.router.seed'])
        if config.contains('support.router.seed') else 62026
    )
    initial_slope = (
        float(config['support.router.initial.slope'])
        if config.contains('support.router.initial.slope') else 1.0
    )
    if mode != 'monotonic_residual':
        raise ValueError('support.router.mode must be monotonic_residual.')
    if not 0.0 < pseudo_cold_ratio < 1.0:
        raise ValueError(
            'support.router.pseudo.cold.ratio must be between 0 and 1.'
        )
    if initial_slope <= 0:
        raise ValueError('support.router.initial.slope must be positive.')
    return {
        'enabled': enabled,
        'mode': mode,
        'pseudo_cold_ratio': pseudo_cold_ratio,
        'seed': seed,
        'initial_slope': initial_slope,
    }


def resolve_hyperedge_attention(config):
    enabled = _config_bool(config, 'hyperedge.attention', False)
    mode = (
        str(config['hyperedge.attention.mode']).strip().lower()
        if config.contains('hyperedge.attention.mode')
        else 'factorized_specificity'
    )
    hc_enabled = _config_bool(config, 'hyperedge.attention.hc', True)
    pd_enabled = _config_bool(config, 'hyperedge.attention.pd', True)
    temperature = (
        float(config['hyperedge.attention.temperature'])
        if config.contains('hyperedge.attention.temperature') else 1.0
    )
    prior_scale = (
        float(config['hyperedge.attention.prior.scale'])
        if config.contains('hyperedge.attention.prior.scale') else 0.1
    )
    if mode != 'factorized_specificity':
        raise ValueError(
            'hyperedge.attention.mode must be factorized_specificity.'
        )
    if enabled and not (hc_enabled or pd_enabled):
        raise ValueError(
            'At least one of hyperedge.attention.hc/pd must be enabled.'
        )
    if temperature <= 0:
        raise ValueError('hyperedge.attention.temperature must be positive.')
    if prior_scale < 0:
        raise ValueError(
            'hyperedge.attention.prior.scale cannot be negative.'
        )
    return {
        'enabled': enabled,
        'mode': mode,
        'hc_enabled': enabled and hc_enabled,
        'pd_enabled': enabled and pd_enabled,
        'temperature': temperature,
        'prior_scale': prior_scale,
    }


def counterfactual_margin_values(
        factual_context_logits,
        counterfactual_context_logits,
        labels,
        eligible_mask,
        margin):
    factual = np.asarray(factual_context_logits, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_context_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=np.float64)
    if not (
            factual.shape == counterfactual.shape == labels.shape
            == eligible.shape):
        raise ValueError('Counterfactual margin inputs must have matching shapes.')
    active = (labels > 0.5).astype(np.float64) * (eligible > 0).astype(np.float64)
    values = np.maximum(
        0.0, float(margin) - (factual - counterfactual)
    ) * active
    return values


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


def resolve_negative_sampling(config):
    strategy = (
        str(config['negative.strategy']).strip().lower()
        if config.contains('negative.strategy') else 'random'
    )
    if strategy not in {'random', 'mixed'}:
        raise ValueError('negative.strategy must be random or mixed.')
    hard_ratio = (
        float(config['negative.hard.ratio'])
        if config.contains('negative.hard.ratio') else 0.25
    )
    if not 0.0 <= hard_ratio <= 1.0:
        raise ValueError('negative.hard.ratio must be between 0 and 1.')
    if strategy == 'mixed' and hard_ratio == 0.0:
        raise ValueError('negative.hard.ratio must be positive for mixed sampling.')
    return {
        'strategy': strategy,
        'hard_ratio': hard_ratio,
    }


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
        decoder_weights=None,
        pair_compound_contexts=None,
        residual_compound_contexts=None,
        target_residual_weight=None,
        herb_protein_scale=None):
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')

    compounds = np.asarray(compound_embeddings)[compound_indices]
    proteins = np.asarray(protein_embeddings)[protein_indices]
    static_herb_contexts = np.asarray(compound_contexts)[compound_indices]
    herb_contexts = (
        np.asarray(pair_compound_contexts)
        if pair_compound_contexts is not None
        else static_herb_contexts
    )
    if herb_contexts.shape != compounds.shape:
        raise ValueError('Pair compound contexts must match the pair embedding shape.')
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
        herb_protein_scores = np.sum(
            herb_contexts * proteins * herb_protein_weight, axis=1
        )
        if herb_protein_scale is not None:
            herb_protein_scale = np.asarray(
                herb_protein_scale, dtype=np.float64
            ).reshape(-1)
            if herb_protein_scale.shape != herb_protein_scores.shape:
                raise ValueError(
                    'Herb-protein scales must match the number of pairs.'
                )
            herb_protein_scores *= herb_protein_scale
        scores += herb_protein_scores
        if residual_compound_contexts is not None:
            residual_contexts = np.asarray(residual_compound_contexts)
            if residual_contexts.shape != compounds.shape:
                raise ValueError(
                    'Residual compound contexts must match the pair embedding shape.'
                )
            if target_residual_weight is None:
                raise ValueError(
                    'Target residual weight is required with residual contexts.'
                )
            context_delta = residual_contexts - static_herb_contexts
            scores += np.sum(
                context_delta * proteins * np.asarray(target_residual_weight), axis=1
            )
    if enabled_terms.get('herb_disease', False):
        scores += np.sum(herb_contexts * disease_contexts * herb_disease_weight, axis=1)
    return scores


def context_masked_pair_scores(
        compound_embeddings,
        protein_embeddings,
        compound_contexts,
        protein_contexts,
        compound_indices,
        protein_indices,
        compound_disease_weight,
        herb_protein_weight,
        herb_disease_weight,
        mask_compound=False,
        mask_protein=False,
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
    effective_compounds = herb_contexts if mask_compound else compounds
    effective_proteins = disease_contexts if mask_protein else proteins
    if enabled_terms is None:
        enabled_terms = {
            'compound_disease': True,
            'herb_protein': True,
            'herb_disease': True,
        }
    scores = pair_decoder_scores(
        effective_compounds,
        effective_proteins,
        decoder_type=decoder_type,
        decoder_weights=decoder_weights,
    )
    if enabled_terms.get('compound_disease', False):
        scores += np.sum(
            effective_compounds * disease_contexts * compound_disease_weight,
            axis=1,
        )
    if enabled_terms.get('herb_protein', False):
        scores += np.sum(
            herb_contexts * effective_proteins * herb_protein_weight,
            axis=1,
        )
    if enabled_terms.get('herb_disease', False):
        scores += np.sum(
            herb_contexts * disease_contexts * herb_disease_weight,
            axis=1,
        )
    return scores


def target_conditioned_herb_contexts(
        herb_edge_embeddings,
        compound_herb_indices,
        compound_herb_mask,
        protein_embeddings,
        compound_indices,
        protein_indices,
        herb_projection,
        protein_projection,
        temperature=1.0):
    herb_edges = np.asarray(herb_edge_embeddings)
    incidence = np.asarray(compound_herb_indices, dtype=np.int64)
    incidence_mask = np.asarray(compound_herb_mask, dtype=np.float32)
    proteins = np.asarray(protein_embeddings)
    compound_indices = np.asarray(compound_indices, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    herb_projection = np.asarray(herb_projection)
    protein_projection = np.asarray(protein_projection)
    temperature = float(temperature)

    if temperature <= 0:
        raise ValueError('Target-attention temperature must be positive.')
    if herb_edges.ndim != 2 or proteins.ndim != 2:
        raise ValueError('Herb-edge and protein embeddings must be rank-2 arrays.')
    if incidence.shape != incidence_mask.shape or incidence.ndim != 2:
        raise ValueError('Compound-herb indices and mask must be matching rank-2 arrays.')
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein index arrays must have the same shape.')
    dimension = herb_edges.shape[1]
    if proteins.shape[1] != dimension:
        raise ValueError('Herb-edge and protein embedding dimensions must match.')
    if herb_projection.shape != (dimension, dimension):
        raise ValueError('Herb projection must have shape [dimension, dimension].')
    if protein_projection.shape != (dimension, dimension):
        raise ValueError('Protein projection must have shape [dimension, dimension].')

    sentinel_index = herb_edges.shape[0]
    if np.any(incidence < 0) or np.any(incidence > sentinel_index):
        raise ValueError('Compound-herb incidence contains an invalid herb index.')
    padded_herb_edges = np.concatenate(
        [herb_edges, np.zeros((1, dimension), dtype=herb_edges.dtype)], axis=0
    )
    pair_incidence = incidence[compound_indices]
    pair_mask = incidence_mask[compound_indices]
    pair_herb_edges = padded_herb_edges[pair_incidence]
    herb_keys = np.matmul(pair_herb_edges, herb_projection)
    protein_queries = np.matmul(proteins[protein_indices], protein_projection)
    logits = np.sum(herb_keys * protein_queries[:, None, :], axis=2)
    logits = logits / (np.sqrt(float(dimension)) * temperature)
    masked_logits = np.where(pair_mask > 0, logits, -1e9)
    shifted_logits = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    unnormalized = np.exp(shifted_logits) * pair_mask
    normalizer = np.sum(unnormalized, axis=1, keepdims=True)
    attention = np.divide(
        unnormalized,
        normalizer,
        out=np.zeros_like(unnormalized),
        where=normalizer > 0,
    )
    contexts = np.sum(attention[:, :, None] * pair_herb_edges, axis=1)
    norms = np.linalg.norm(contexts, axis=1, keepdims=True)
    contexts = np.divide(
        contexts,
        norms,
        out=np.zeros_like(contexts),
        where=norms > 0,
    )
    return contexts, attention


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
