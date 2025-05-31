import argparse
import dill
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import CHGRec
from model.causal_construction import CausaltyGraph4Visit
from model.test import Test
from model.util import buildPrjSmiles


def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)


def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument("--debug", default=False,
                        help="debug mode, the number of samples, "
                             "the number of generations run are very small, "
                             "designed to run on cpu, the development of the use of")
    parser.add_argument("--Test", default=False, help="test mode")

    # environment
    parser.add_argument('--dataset', default='mimic3', help='mimic3/mimic4')
    parser.add_argument('--resume_path', default="../data/mimic3/best_model.pkl", type=str,
                        help='path of well trained model, only for evaluating the model, needs to be replaced manually')
    parser.add_argument('--device', type=int, default=0, help='gpu id to run on, negative for cpu')

    # parameters
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument("--regular", type=float, default=0.005, help="regularization parameter")
    parser.add_argument('--target_ddi', type=float, default=0.06, help='expected ddi for training')
    parser.add_argument('--coef', default=2.5, type=float, help='coefficient for DDI Loss Weight Annealing')
    parser.add_argument('--epochs', default=25, type=int, help='the epochs for training')

    args = parser.parse_args()


    return args


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device("cpu")
        if not args.Test:
            print("GPU unavailable, switch to debug mode")
            args.debug = True
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = f'../data/{args.dataset}/records_final.pkl'
    voc_path = f'../data/{args.dataset}/voc_final.pkl'
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ddi_adj_path = f'../data/{args.dataset}/ddi_A_final.pkl'
    ddi_mask_path = f'../data/{args.dataset}/ddi_mask_H.pkl'
    molecule_path = f'../data/{args.dataset}/idx2drug.pkl'
    relevance_diag_med_path = f'../data/{args.dataset}/Diag_Med_causal_effect.pkl'
    relevance_proc_med_path = f'../data/{args.dataset}/Proc_Med_causal_effect.pkl'
    ehr_adj_path = f'../data/{args.dataset}/ehr_adj_final.pkl'


    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
        if args.debug:
            data = data[:5]
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(relevance_proc_med_path, 'rb') as Fin:
        relevance_proc_med = dill.load(Fin)
    with open(relevance_diag_med_path, 'rb') as Fin:
        relevance_diag_med = dill.load(Fin)
    with open(ehr_adj_path, 'rb') as Fin:
        ehr_adj = torch.from_numpy(dill.load(Fin)).to(device)



    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    ]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    binary_projection, average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)

    relevance_diag_mole = np.dot(relevance_diag_med.to_numpy(), binary_projection)
    relevance_proc_mole = np.dot(relevance_proc_med.to_numpy(), binary_projection)
    relevance_med_mole = average_projection
    mole_relevance = [relevance_diag_mole, relevance_proc_mole, relevance_med_mole, binary_projection, relevance_diag_med, relevance_proc_med]
    voc_size.append(relevance_med_mole.shape[1])

    causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], args.dataset)

    model = CHGRec(
        causal_graph=causal_graph,
        mole_relevance=mole_relevance,
        tensor_ddi_adj=ddi_adj,
        dropout=args.dp,
        emb_dim=args.dim,
        voc_size=voc_size,
        ehr_adj=ehr_adj,
        device=device
    ).to(device)


    if args.Test:

        model.load_state_dict(torch.load(open(f'../data/{args.dataset}/best_model.pkl', 'rb')))
        Test(model, device, data_test, voc_size, args)

