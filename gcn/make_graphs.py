# EXPECTS PREPROCESSED PT FILES TO BE ON DISK
# reads the RoBERTa pre-processed .pt files from disk
# and saves graphs, gcn_offsets & cls_offsets for model use.
import spacy
from spacy import displacy
import en_core_web_lg
import pprint
import collections

import re

import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset
import dgl.function as fn
from dgl.data.utils import save_graphs

import torch

import pandas as pd

import numpy as np

parser = en_core_web_lg.load()

def run():
    for size in ['xl']:
        print("Loading data...")
        X_preprocessed = torch.load("data/bert_preprocessed/X_{}.pt".format(size)) # load BERT pre-processed data from disk
        y_data = torch.load("data/bert_preprocessed/y_{}.pt".format(size))
        print("Done loading.\n")

        y_outputs = []

        all_graphs, gcn_offsets, cls_tokens, skip_indices = convert_preprocessed_rows_to_graph(X_preprocessed)

        print("Skipped indices:", len(skip_indices))

        for y_idx, y in enumerate(y_data):
            if y_idx in skip_indices:
                continue
            y_outputs.append(y)

        cls_tokens = torch.stack(cls_tokens)
        y_outputs  = torch.stack(y_outputs)
        gcn_offsets = torch.tensor(gcn_offsets)

        # https://docs.dgl.ai/en/0.4.x/generated/dgl.data.utils.load_graphs.html
        save_graphs("data/Jack_X_train_graphs_{}.bin".format(size), all_graphs)
        torch.save(cls_tokens, "data/Jack_X_train_cls_tokens_{}.bin".format(size))
        torch.save(gcn_offsets, "data/Jack_X_train_gcn_offsets_{}.bin".format(size))
        torch.save(y_outputs, "data/Jack_y_train_{}.bin".format(size))

### HELPER FUNCTIONS ####

def convert_preprocessed_rows_to_graph(rows):
    all_graphs = []
    gcn_offsets = []
    cls_tokens = []
    skip_indices = []
    broken_examples = 0

    for row_idx, row in enumerate(rows):
        processed_output = convert_preprocessed_row_to_graph(row)

        if processed_output is None:
            skip_indices.append(row_idx)
            broken_examples += 1
            continue

        graph, gcn_offset, cls_token = processed_output
        all_graphs.append(graph)
        gcn_offsets.append(gcn_offset)
        cls_tokens.append(cls_token)

        if row_idx % 100 == 99:
            print("Finished row {} out of {}".format(row_idx+1, len(rows)))

    print("Number of broken examples:", broken_examples)
    return all_graphs, gcn_offsets, cls_tokens, skip_indices

def convert_preprocessed_row_to_graph(row):
    sentence = row['sentence']
    bert_embeddings = row['encoding'][0]
    bert_tokens = row['tokens']
    options = row['options']
    parsed_options = (parser(options[0]), parser(options[1]))

    doc = parser(sentence)
    nodes = collections.OrderedDict()
    edges = []
    edge_type = []

    offsets = []
    offset_words = []

    spacy_tokens = []

    trial_offsets = []
    for token in doc:
        if not is_target(token, options, parsed_options):
            continue

        trial_offsets.append(token.i)

    if len(trial_offsets) < 3:
        return None

    for token in doc:

        spacy_tokens.append(token)

        # skip words that aren't targets or separated by one edge from target
        if not (is_target(token, options, parsed_options) or is_target(token, options, parsed_options)):
            continue

        if token.i not in nodes:
            nodes[token.i] = len(nodes)
            edges.append( [token.i, token.i])
            edge_type.append(0)

        if token.head.i not in nodes:
            nodes[token.head.i] = len(nodes)
            edges.append( [token.head.i, token.head.i] )
            edge_type.append(0)

        if token.dep_ != 'ROOT':
            edges.append( [ token.head.i, token.i ])
            edge_type.append(1)
            edges.append( [ token.i, token.head.i ])
            edge_type.append(2)

        if is_target(token, options, parsed_options):
            if len(offsets) == 3:
                offsets[2] = token.i
                offset_words.append(token.text)
            else:
                offsets.append(token.i)
                offset_words.append(token.text)

    num_nodes, tran_edges = transfer_n_e(nodes, edges)

    if (len(offsets) != 3):
        print("UNEXPECTED: at least 3 positions should be in offsets")
        print(sentence, options, len(offsets))
        print(offset_words)

    gcn_offset = [nodes[offset] for offset in offsets]

    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)
    G.add_edges(list(zip(*tran_edges))[0], list(zip(*tran_edges))[1])

    # Transform tokens and head tokens into bert contractions
    db_tokens = []
    db_head_tokens = []
    for token in doc:
        db_tokens.append(str(token))
        db_head_tokens.append(str(token.head))

    contractions = []
    for i in range(len(db_tokens)):
        if db_tokens[i] == "n\'t" or db_tokens[i] == "n’t":
            db_tokens[i] = "\'t"
            contractions.append(db_tokens[i-1])
            db_tokens[i-1] = db_tokens[i-1] + "n"

    for i in range(len(db_head_tokens)):
        if db_head_tokens[i] == "n\'t" or db_head_tokens[i] == "n’t":
            db_head_tokens[i] = "\'t"
        if db_head_tokens[i] in contractions:
            db_head_tokens[i] = db_head_tokens[i] + "n"

    idx = 0
    for token in doc:
        if not (is_target(token, options, parsed_options) or is_target(token, options, parsed_options)):
            continue

        embedding = bert_embedding_for_dp_token(db_tokens[idx], bert_tokens, bert_embeddings)
        if db_tokens[idx] == "\'t":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('t', bert_tokens, bert_embeddings)
        elif db_tokens[idx] == "can":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('cannot', bert_tokens, bert_embeddings)
        elif db_tokens[idx] == "not":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('cannot', bert_tokens, bert_embeddings)
        elif db_tokens[idx] == "wo":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('wont', bert_tokens, bert_embeddings)
        elif db_tokens[idx] == "nt":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('wont', bert_tokens, bert_embeddings)
        
        if(torch.isnan(embedding.unsqueeze(0)).any()):
            print("UNEXPECTED (A): bert_embedding_for_dp_token returns NaN - ", db_tokens[idx])
            # print(embedding.unsqueeze(0))
            # print(token.i, token.text, sentence, bert_tokens)
            # print(spacy_tokens)
            print(sentence)
            print(spacy_tokens)
            print(bert_tokens)
            print(bert_embeddings)
            print('\n')
            G.nodes[ nodes[token.i] ].data['h'] = torch.randn(1024).unsqueeze(0)
        else:
            G.nodes[ nodes[token.i] ].data['h'] = embedding.unsqueeze(0)

        embedding = bert_embedding_for_dp_token(db_head_tokens[idx], bert_tokens, bert_embeddings)
        if db_head_tokens[idx] == "\'t":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('t', bert_tokens, bert_embeddings)
        elif db_head_tokens[idx] == "can":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('cannot', bert_tokens, bert_embeddings)
        elif db_head_tokens[idx] == "not":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('cannot', bert_tokens, bert_embeddings)
        elif db_head_tokens[idx] == "wo":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('wont', bert_tokens, bert_embeddings)
        elif db_head_tokens[idx] == "nt":
            if(torch.isnan(embedding.unsqueeze(0)).any()):
                embedding = bert_embedding_for_dp_token('wont', bert_tokens, bert_embeddings)

        if(torch.isnan(embedding.unsqueeze(0)).any()):
            print("UNEXPECTED (B): bert_embedding_for_dp_token returns NaN - ", db_head_tokens[idx])
            # print(embedding.unsqueeze(0))
            # print(token.i, token.head.i, token.head.text, sentence, bert_tokens)
            # print(spacy_tokens)
            print(sentence)
            print(spacy_tokens)
            print(bert_tokens)
            print('\n')
            G.nodes[ nodes[token.head.i] ].data['h'] = torch.randn(1024).unsqueeze(0)
        else:
            G.nodes[ nodes[token.head.i] ].data['h'] = embedding.unsqueeze(0)
        
        idx += 1

    edge_norm = []
    for e1, e2 in tran_edges:
        if e1 == e2:
            edge_norm.append(1)
        else:
            edge_norm.append( 1 / (G.in_degree(e2) - 1 ) )

    edge_type = torch.from_numpy(np.array(edge_type))
    edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float()

    G.edata.update({'rel_type': edge_type,})
    G.edata.update({'norm': edge_norm})

    return G, gcn_offset, bert_embeddings[0]

def is_target(dp_token, options, parsed_options, debug=False):

    option0 = re.split(' |-', options[0])[0]
    option1 = re.split(' |-', options[1])[0]

    option0_lemmatized = option0
    option1_lemmatized = option1
    dp_token_lemmatized = dp_token

    if dp_token.text == option0:
        return True

    if dp_token.text == option1:
        return True

    if (dp_token.lemma_.lower() == parsed_options[0][0].lemma_.lower()):
        return True

    if (dp_token.lemma_.lower() == parsed_options[1][0].lemma_.lower()):
        return True

    if ((dp_token.pos_ == "PROPN" or dp_token.pos_ == "NOUN") and dp_token.text[-1] == 's'):
        if (dp_token.text[:-1].lower() == parsed_options[0][0].lemma_.lower()):
            return True
        if (dp_token.text[:-1].lower() == parsed_options[1][0].lemma_.lower()):
            return True

    return False

def bert_embedding_for_dp_token(token, bert_tokens, bert_embeddings, debug=False):
    try:
        idx = bert_tokens.index(token)
        return bert_embeddings[idx]
    except ValueError:
        temp = token
        start, end = 0, 0
        seq = False
        for i, bert_token in enumerate(bert_tokens):
            if debug:
                print("HA:", bert_token, token)
            if bert_token in temp and temp.index(bert_token) == 0:
                temp = temp[len(bert_token):]
                if not seq:
                    start = i
                    seq = True
            else:
                temp = token
                seq = False
                if bert_token in temp and temp.index(bert_token) == 0:
                    temp = temp[len(bert_token):]
                    if not seq:
                        start = i
                        seq = True
            if len(temp) == 0:
                end = i + 1
                break
        
        if (debug):
            print(start, end)

        bert_emb_tensor = bert_embeddings[start:end]
        return torch.mean(bert_emb_tensor, dim=0)

def transfer_n_e(nodes, edges):

    num_nodes = len(nodes)
    new_edges = []
    for e1, e2 in edges:
        new_edges.append( [nodes[e1], nodes[e2]] )
    return num_nodes, new_edges

run()