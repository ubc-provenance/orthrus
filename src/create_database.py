from create_database import cadets_e5, cadets_e3, clearscope_e5, clearscope_e3, theia_e3, theia_e5
from config import *
from provnet_utils import *

def main(cfg):
    dataset = cfg.dataset.name
    if dataset == 'CADETS_E3':
        cadets_e3.main(cfg)
    elif dataset == 'CADETS_E5':
        cadets_e5.main(cfg)
    elif dataset == 'THEIA_E3':
        theia_e3.main(cfg)
    elif dataset == 'THEIA_E5':
        theia_e5.main(cfg)
    elif dataset == 'CLEARSCOPE_E3':
        clearscope_e3.main(cfg)
    elif dataset == 'CLEARSCOPE_E5':
        clearscope_e5.main(cfg)

if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    cfg = get_yml_cfg(args)
    main(cfg)