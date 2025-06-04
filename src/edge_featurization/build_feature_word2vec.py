import os
from provnet_utils import *
from config import *
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np
import random
import torch

def load_corpus_from_database(indexid2msg, use_node_types):

    corpus = {}
    for indexid, msg in tqdm(indexid2msg.items(), desc='Tokenizing corpus from database:'):
        if msg[0] == 'subject':
            if use_node_types:
                tokens = tokenize_subject(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_subject(msg[1])
        elif msg[0] == 'file':
            if use_node_types:
                tokens = tokenize_file(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_file(msg[1])
        else:
            if use_node_types:
                tokens = tokenize_netflow(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_netflow(msg[1])
        corpus[msg[1]] = tokens
    return list(corpus.values())

def train_feature_word2vec(corpus, cfg, model_save_path, logger):
    emb_dim = cfg.edge_featurization.embed_nodes.emb_dim
    show_epoch_loss = cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss
    window_size = cfg.edge_featurization.embed_nodes.feature_word2vec.window_size
    min_count = cfg.edge_featurization.embed_nodes.feature_word2vec.min_count
    use_skip_gram = cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram
    num_workers = cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers
    epochs = cfg.edge_featurization.embed_nodes.feature_word2vec.epochs
    compute_loss = cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss
    negative = cfg.edge_featurization.embed_nodes.feature_word2vec.negative
    use_seed = cfg.edge_featurization.embed_nodes.use_seed
    SEED = cfg._seed

    if show_epoch_loss:
        if use_seed:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=1,
                             compute_loss=compute_loss,
                             negative=negative,
                             seed=SEED)
        else:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=1,
                             compute_loss=compute_loss,
                             negative=negative)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: 0/{epochs}; loss: {epoch_loss}")

        for epoch in range(epochs - 1):
            model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
            epoch_loss = model.get_latest_training_loss()
            log(f"Epoch: {epoch+1}/{epochs}; loss: {epoch_loss}")
    else:
        if use_seed:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=epochs,
                             compute_loss=compute_loss,
                             negative=negative,
                             seed=SEED)
        else:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=epochs,
                             compute_loss=compute_loss,
                             negative=negative)
        loss = model.get_latest_training_loss()
        log(f"Epoch: {epochs}; loss: {loss}")

    model.init_sims(replace=True)
    model.save(os.path.join(model_save_path, 'feature_word2vec.model'))
    log(f"Save word2vec to {os.path.join(model_save_path, 'feature_word2vec.model')}")

def main(cfg):
    model_save_path = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    logger = get_logger(
        name="build_feature_word2vec",
        filename=os.path.join(cfg.edge_featurization.embed_nodes._logs_dir, "feature_word2vec.log")
    )
    log(f"Building feature word2vec model and save model to {model_save_path}")

    use_node_types = cfg.edge_featurization.embed_nodes.feature_word2vec.use_node_types
    use_cmd =  cfg.edge_featurization.embed_nodes.feature_word2vec.use_cmd
    use_port = cfg.edge_featurization.embed_nodes.feature_word2vec.use_port
    use_seed = cfg.edge_featurization.embed_nodes.use_seed

    if use_seed:
        SEED = cfg._seed
        np.random.seed(SEED)
        random.seed(SEED)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log(f"Get indexid2msg from database...")
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur, use_cmd=use_cmd, use_port=use_port)

    log(f"Start building and training feature word2vec model...")

    log("Loading and tokenizing corpus from database...")
    corpus = load_corpus_from_database(indexid2msg=indexid2msg, use_node_types=use_node_types)

    train_feature_word2vec(corpus=corpus,
                           cfg=cfg,
                           model_save_path=model_save_path,
                           logger=logger)

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)