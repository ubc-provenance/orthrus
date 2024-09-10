import argparse
import os
import hashlib
import pathlib
import sys
import yaml
from copy import deepcopy
from collections import OrderedDict
from pprint import pprint
from yacs.config import CfgNode as CN
from psycopg2 import extras as ex
import psycopg2

# [EDITABLE AREA]: Insert your output path and credentials to the DB
# ================================================================================
ROOT_ARTIFACT_DIR = "/data1/tbilot/orthrus/" # Destination folder for generated files. Will be created if doesn't exist.
ROOT_GROUND_TRUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ground_Truth/")


DATABASE_DEFAULT_CONFIG = {
     "host": '',  # Host machine where the db is located
     "user": '',  # Database user
     "password": '',  # The password to the database user
     "port": '',  # The port number for Postgres
}
# ================================================================================

# --- Dependency graph to follow ---
TASK_DEPENDENCIES = OrderedDict({
     "build_graphs": [],
     "embed_nodes": ["build_graphs"],
     "embed_edges": ["embed_nodes"],
     "gnn_training": ["embed_edges"],
     "gnn_testing": ["gnn_training"],
     "evaluation": ["gnn_testing"],
     "tracing" : ["evaluation"],
})

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
     "preprocessing": {
          "build_graphs": {
               "used_method": str, # [orthrus | magic]
               "use_all_files": bool,
               "time_window_size": float,
               "use_hashed_label": bool,
               "node_label_features": {
                    "subject": str,  # [type, path, cmd_line]
                    "file": str,  # [type, path]
                    "netflow": str,  # [type, remote_ip, remote_port]
               },
          },
     },
     "featurization": {
          "embed_nodes": {
               "emb_dim": int,
               "used_method": str,
               "use_seed": bool,
               "feature_word2vec": {
                    'show_epoch_loss': bool,
                    'window_size': int,
                    'min_count': int,
                    'use_skip_gram': bool,
                    'num_workers': int,
                    'epochs': int,
                    'compute_loss': bool,
                    'negative': int,
                    'use_node_types': bool,
                    'use_cmd': bool,
                    'use_port': bool,
                    'decline_rate': int,
               },
          },
          "embed_edges": {
               "to_remove": bool, # TODO: remove
          }
     },
     "detection": {
          "gnn_training": {
               "used_method": str, # [ "magic" | "orthrus" | "flash" ]
               "use_seed": bool,
               "num_epochs": int,
               "lr": float,
               "weight_decay": float,
               "node_hid_dim": int,
               "node_out_dim": int,
               "encoder": {
                    "use_node_type_in_node_feats": bool,
                    "neighbor_sampling": list,  # [[] | [int, ..., int]]
                    "edge_features": str,  # ["edge_type", "msg", "time_encoding", "none"]
                    "use_node_feats_in_gnn": bool,
                    "neighbor_size": int,
                    "temporal_dim": int,
                    "batch_size": int,
                    "graph_attention": {
                         "dropout": float,
                         "activation": str,
                         "num_heads": int,
                    },
               },
               "decoder": {
                    "used_methods": str,
                    "predict_edge_type": {
                         "used_method": str,  # ["custom"]
                         "custom": {
                              "dropout": float,
                              "num_layers": int,  # [2 | 3]
                              "activation": str,  # ["sigmoid" | "tanh" | "relu" | "none"]
                         }
                    },
               },
          },
          "gnn_testing": {
               "threshold_method": str,
          },
          "evaluation": {
               "viz_malicious_nodes": bool,
               "ground_truth_version": str,  # ["darpa_v4"]
               "used_method": str,
               "node_evaluation": {
                    "threshold_method": str,  # ["max_val_loss" | "mean_val_loss"]
                    "use_dst_node_loss": bool,
                    "use_kmeans": bool,
                    "kmeans_top_K": int,
               },
          },
     },
     "triage": {
          "tracing": {
               "used_method": str, #["depimpact"]
               "depimpact": {
                    "used_method": str, #["component" | "shortest_path" | "1-hop" | "2-hop" | "3-hop"]
                    "score_method": str, # ["degree" | "recon_loss" | "degree_recon"]
                    "workers": int,
                    "visualize" : bool,
               },
          },
     },
     "postprocessing":
          {},
}

DATASET_DEFAULT_CONFIG = {
     "THEIA_E5": {
          "raw_dir": "",
          "database": "theia_e5_new",
          "database_all_file": "theia_e5_new",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15"],
          "unused_files": ["graph_12", "graph_13", "graph_16", "graph_17"],
          "ground_truth_relative_path": ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"],
          "attack_to_time_window" : [
               ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv" , '2019-05-15 14:47:00', '2019-05-15 15:08:00'],
          ]
     },
     "THEIA_E3": {
          "raw_dir": "",
          "database": "theia_e3",
          "database_all_file": "theia_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_2", "graph_3", "graph_4", "graph_5", "graph_6", "graph_7", "graph_8"],
          "val_files": ["graph_9"],
          "test_files": ["graph_10", "graph_12", "graph_13"],
          "unused_files": ["graph_11"],
          "ground_truth_relative_path": ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                                         "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                                         # "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it
                                         # "E3-THEIA/node_Phishing_E_mail_Link.csv" # attack only at network level, not system
                                         ],
          "attack_to_time_window" : [
               ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv" , '2018-04-12 12:40:00', '2018-04-12 13:30:00'],
               ["E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv" , '2018-04-10 14:30:00', '2018-04-10 15:00:00'],
          ]
     },
     "CADETS_E5": {
          "raw_dir": "/media/bxjiang/Expansion/dataset/cadets_e5/json_cadets/",
          "database": "cadets_e5",
          "database_all_file": "cadets_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_11"],
          "val_files": ["graph_12"],
          "test_files": ["graph_16", "graph_17"],
          "unused_files": ["graph_15", "graph_10", "graph_13", "graph_14"],
          "ground_truth_relative_path": ["E5-CADETS/node_Nginx_Drakon_APT.csv",
                                         "E5-CADETS/node_Nginx_Drakon_APT_17.csv"],
          "attack_to_time_window" : [
               ["E5-CADETS/node_Nginx_Drakon_APT.csv" , '2019-05-16 09:31:00', '2019-05-16 10:12:00'],
               ["E5-CADETS/node_Nginx_Drakon_APT_17.csv" , '2019-05-17 10:15:00', '2019-05-17 15:33:00'],
          ]
     },
     "CADETS_E3": {
          "raw_dir": "",
          "database": "cadets_e3",
          "database_all_file": "cadets_e3_all",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_2"],
          "test_files": ["graph_6", "graph_11", "graph_12", "graph_13"],
          "unused_files": [],
          "ground_truth_relative_path": [
                                         "E3-CADETS/node_Nginx_Backdoor_06.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_12.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_13.csv"],
          "attack_to_time_window": [
               ["E3-CADETS/node_Nginx_Backdoor_06.csv" , '2018-04-06 11:20:00', '2018-04-06 12:09:00'],
               ["E3-CADETS/node_Nginx_Backdoor_12.csv" , '2018-04-12 13:59:00', '2018-04-12 14:39:00'],
               ["E3-CADETS/node_Nginx_Backdoor_13.csv" , '2018-04-13 09:03:00', '2018-04-13 09:16:00'],
          ],
     },
     "CLEARSCOPE_E5": {
          "raw_dir": "",
          "database": "clearscope_e5",
          "database_all_file": "clearscope_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15", "graph_17"],
          "unused_files": ["graph_10", "graph_12", "graph_13", "graph_16"],
          "ground_truth_relative_path": [
               "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
               "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
               "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
          ],
          "attack_to_time_window": [
               ["E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv", '2019-05-15 15:38:00', '2019-05-15 16:19:00'],
               ["E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv", '2019-05-17 15:48:00', '2019-05-17 16:01:00'],
               ["E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv", '2019-05-17 16:20:00', '2019-05-17 16:28:00'],
          ],
     },
     "CLEARSCOPE_E3": {
          "raw_dir": "",
          "database": "clearscope_e3",
          "database_all_file": "clearscope_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_2"],
          "test_files": ["graph_11", "graph_12"],
          "unused_files": ["graph_6", "graph_13"],
          "ground_truth_relative_path": [
               "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
               # "E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", # due to malicious file downloaded but failed to exec and feture missing, there is no malicious nodes found in database
          ],
          "attack_to_time_window": [
               ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", '2018-04-11 13:54:00', '2018-04-11 14:48:00'],
               # ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", '2018-04-12 15:18:00', '2018-04-12 15:25:00'],
          ],
     },
}

def get_default_cfg(args):
     """
     Inits the shared cfg object with default configurations.
     """
     cfg = CN()
     cfg._artifact_dir = ROOT_ARTIFACT_DIR

     cfg._test_mode = False

     cfg._use_cpu = args.cpu

     # Database: we simply create variables for all configurations described in the dict
     cfg.database = CN()
     for attr, value in DATABASE_DEFAULT_CONFIG.items():
          setattr(cfg.database, attr, value)

     # Dataset: we simply create variables for all configurations described in the dict
     cfg.dataset = CN()
     cfg.dataset.name = args.dataset
     for attr, value in DATASET_DEFAULT_CONFIG[cfg.dataset.name].items():
          setattr(cfg.dataset, attr, value)
     
     # Tasks: we create nested None variables for all arguments
     def create_cfg_recursive(cfg, task_args_dict: dict):
          for task, subtasks in task_args_dict.items():
               if isinstance(subtasks, dict):
                    setattr(cfg, task, CN())
                    task_cfg = getattr(cfg, task)
                    create_cfg_recursive(task_cfg, dict(subtasks.items()))
               else:
                    setattr(cfg, task, None)

     create_cfg_recursive(cfg, TASK_ARGS)
     
     return cfg

def get_runtime_required_args(return_unknown_args=False, args=None):
     parser = argparse.ArgumentParser()
     parser.add_argument('dataset', type=str, help="Name of the dataset")
     parser.add_argument('--model', type=str, help="Name of the model (Orthrus)")
     parser.add_argument('--wandb', action="store_true", help="Whether to submit logs to wandb")
     parser.add_argument('--exp', type=str, default="", help="Name of the experiment")
     parser.add_argument('--tags', type=str, default="", help="Name of the tag to use. Tags are used to group runs together")
     parser.add_argument('--cpu', action="store_true", help="Whether to run on CPU rather than GPU")
     parser.add_argument('--run_from_training', action="store_true", help="Runs Orthrus from training when graphs are arleady preprocessed")

     parser.add_argument('--show_attack', type=int, help="Number of attack for plotting", default=0)
     parser.add_argument('--gt_type', type=str, help="Type of ground truth", default="orthrus")
     parser.add_argument('--plot_gt', type=bool, help="If we plot ground truth", default=False)

     # All args in the cfg can be also set in the arg parser from CLI
     parser = add_cfg_args_to_parser(TASK_ARGS, parser)
     
     try:
          args, unknown_args = parser.parse_known_args(args)
     except:
          parser.print_help()
          sys.exit(1)

     args.model = "orthrus"
     
     if return_unknown_args:
          return args, unknown_args
     return args

def overwrite_cfg_with_args(cfg, args):
     """
     The framework can be also parametrized using the CLI args.
     These args are priorited compared to yml file parameters.
     This function simply overwrites the cfg with the parameters 
     given within args.
     
     To override a parameter in cfg, use a dotted style:
     ```python orthrus.py --detection.gnn_training.seed=42```
     """
     for arg, value in args.__dict__.items():
          if "." in arg and value is not None:
               cfg_ptr = cfg
               dots = arg.split(".")
               path, attr_name = dots[:-1], dots[-1]
               
               for attr in path:
                    cfg_ptr = getattr(cfg_ptr, attr)
               setattr(cfg_ptr, attr_name, value)

def set_task_paths(cfg):
     subtask_to_hash = {}
     # Directories common to all tasks
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)

          # We first compute a unique hash for each usbtask
          for subtask_name, subtask_args in subtask.items():
               subtask_cfg = getattr(task_cfg, subtask_name)
               restart_values = flatten_arg_values(subtask_cfg)

               clean_hash_args = ["".join([c for c in str(restart_value) if c not in set(" []\"\'")]) for restart_value in restart_values]
               final_hash_string = ",".join(clean_hash_args)
               final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()
               
               subtask_to_hash[subtask_name] = final_hash_string

     # Then, for each subtask, we want its unique hash to also depend from its previous dependencies' hashes.
     # For example, if I run the same subtask A two times, with two different subtasks B and C, the results
     # would be different and would be stored in the same folder A if we don't consider the hash of B and C.
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)
          for subtask_name, subtask_args in subtask.items():
               subtask_cfg = getattr(task_cfg, subtask_name)
               deps = sorted(list(get_dependees(subtask_name, TASK_DEPENDENCIES, set())))
               deps_hash = "".join([subtask_to_hash[dep] for dep in deps])
               
               final_hash_string = deps_hash + subtask_to_hash[subtask_name]
               final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()
               
               if task in ["preprocessing", "featurization"]:
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, cfg.dataset.name, subtask_name, final_hash_string)
               else:
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, subtask_name, final_hash_string, cfg.dataset.name)
               
               # The directory to save logs related to the preprocessing task
               subtask_cfg._logs_dir = os.path.join(subtask_cfg._task_path, "logs/")
               os.makedirs(subtask_cfg._logs_dir, exist_ok=True)
     
     # Preprocessing paths
     cfg.preprocessing.build_graphs._graphs_dir = os.path.join(cfg.preprocessing.build_graphs._task_path, "nx/")
     cfg.preprocessing.build_graphs._tw_labels = os.path.join(cfg.preprocessing.build_graphs._task_path, "tw_labels/")
     cfg.preprocessing.build_graphs._node_id_to_path = os.path.join(cfg.preprocessing.build_graphs._task_path, "node_id_to_path/")

     # Featurization paths
     cfg.featurization.embed_nodes.feature_word2vec._model_dir = os.path.join(cfg.featurization.embed_nodes._task_path, "word2vec_models/")
     cfg.featurization.embed_edges._edge_embeds_dir = os.path.join(cfg.featurization.embed_edges._task_path, "edge_embeds/")

     # Detection paths
     cfg.detection.gnn_training._trained_models_dir = os.path.join(cfg.detection.gnn_training._task_path, "trained_models/")
     cfg.detection.gnn_testing._edge_losses_dir = os.path.join(cfg.detection.gnn_testing._task_path, "edge_losses/")
     cfg.detection.evaluation.node_evaluation._precision_recall_dir = os.path.join(cfg.detection.evaluation._task_path, "precision_recall_dir/") # TODO: move to cfg.detection._precision_recall_dir
     cfg.detection.evaluation._evaluation_results_dir = os.path.join(cfg.detection.evaluation._task_path, "evaluation_results/")

     # Ground Truth paths
     cfg._ground_truth_dir = os.path.join(ROOT_GROUND_TRUTH_DIR, cfg.detection.evaluation.ground_truth_version + '/')

     # Triage paths
     cfg.triage.tracing._tracing_graph_dir = os.path.join(cfg.triage.tracing._task_path, "tracing_graphs")
     
     # TODO
     cfg.postprocessing._task_path = None

def validate_yml_file(yml_file: str):
     with open(yml_file, 'r') as file:
          user_config = yaml.safe_load(file)

     def validate_config(user_config, tasks, path=None):
          if path is None:
               path = []
          if not user_config:
               raise ValueError(f"Config at {' > '.join(path)} is empty but should not be.")

          for key, sub_tasks in tasks.items():
               if key in user_config:
                    sub_config = user_config[key]
                    if isinstance(sub_tasks, dict):
                         # Recursive check for sub-dictionaries
                         validate_config(sub_config, sub_tasks, path + [key])
                    else:
                         # Check for None values in parameters
                         if sub_config is None:
                              raise ValueError(f"Parameter '{' > '.join(path + [key])}' should not be None.")
                              # Optional: check for type correctness
                         if not isinstance(sub_config, sub_tasks):
                              raise TypeError(f"Parameter '{' > '.join(path + [key])}' should be of type {sub_tasks.__name__}.")
     
     validate_config(user_config, TASK_ARGS)
     print(f"YAML configuration file \"{yml_file.split('/')[-1]}\" is valid")

def check_args(args):
     available_models = os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config"))
     if not any([args.model in model for model in available_models]):
          raise ValueError(f"Unknown model {args.model}. Available models are {available_models}")
     
     available_datasets = DATASET_DEFAULT_CONFIG.keys()
     if args.dataset not in available_datasets:
          raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are {available_datasets}")

def check_task_dependency_graph(yml_file: str):
     with open(yml_file, 'r') as file:
          user_config = yaml.safe_load(file)
     
     subtasks = [j for i in user_config.values() for j in i]
     deps = TASK_DEPENDENCIES
     subtask_set = set(subtasks)

     def has_all_dependencies(task):
          return all(dependency in subtask_set and has_all_dependencies(dependency)
               for dependency in deps.get(task, []))

     dependencies_ok = all(has_all_dependencies(subtask) for subtask in subtasks)
     if dependencies_ok:
          print(f"Task dependency graph is valid: {subtasks}")
          # log("\nYAML configuration")
          # log(user_config)
     else:
          raise ValueError(("The requested subtasks don't respect the subtask dependency graph."
               f"Tasks: {subtasks}\nTask dependency graph: {deps}"))

def get_yml_cfg(args):
     # Checks that CLI args are OK
     check_args(args)
     
     # Inits with default configurations
     cfg = get_default_cfg(args)

     # Checks that all configurations are valid (not set to None)
     root_path = pathlib.Path(__file__).parent.parent.resolve()
     yml_file = f"{root_path}/config/{args.model}.yml"
     validate_yml_file(yml_file)

     # Overrides default config with config from yml file
     cfg.merge_from_file(yml_file)
     
     # Overwrites args to the cfg
     overwrite_cfg_with_args(cfg, args)

     # Asserts all required configurations are present in the final cfg
     check_task_dependency_graph(yml_file)

     # Based on the defined restart args, computes a unique path on disk
     # to store the files of each task
     set_task_paths(cfg)

     return cfg

def get_dependencies(sub: str, dependencies: dict, result_set: set):
     """
     Returns the set of the subtasks happening after `sub`.
     """
     def helper(sub):
          for subtask, deps in dependencies.items():
               if sub in deps:
                    result_set.add(subtask)
                    helper(subtask)
     helper(sub)
     return result_set

def get_dependees(sub: str, dependencies: dict, result_set: set):
     """
     Returns the set of the subtasks happening before `sub`.
     """
     dependencies = OrderedDict(sorted(dependencies.items(), reverse=True))

     def helper(sub):
          for subtask, deps in dependencies.items():
               if sub == subtask:
                    if len(deps) > 0:
                         dep = deps[0]
                         result_set.add(dep)
                         helper(dep)
     helper(sub)
     return result_set

def flatten_arg_values(cfg):
     def helper(dict_or_val, flatten_list):
          if isinstance(dict_or_val, dict):
               for key, value in dict_or_val.items():
                    if isinstance(value, dict):
                         helper(value, flatten_list)
                    else:
                         helper(f"{key}={value}", flatten_list)
          else:
               flatten_list.append(dict_or_val)
     
     flatten_list = []
     helper(cfg, flatten_list)
     return flatten_list

def add_cfg_args_to_parser(cfg, parser):
     def str2bool(v):
          if isinstance(v, bool):
               return v
          elif v == "None":
               return None
          if v.lower() in ('true'):
               return True
          elif v.lower() in ('false'):
               return False
          else:
               raise argparse.ArgumentTypeError('Boolean value expected.')

     def nested_dict_to_separator_dict(nested_dict, separator='.'):
          def _create_separator_dict(x, key='', separator_dict={}, keys_to_ignore=[]):
               if isinstance(x, dict):
                    for k, v in x.items():
                         kk = f'{key}{separator}{k}' if key else k
                         _create_separator_dict(x[k], kk, keys_to_ignore=keys_to_ignore)
               else:
                    if not any([ignore in key for ignore in keys_to_ignore]):
                         separator_dict[key] = x
               return separator_dict

          return _create_separator_dict(deepcopy(nested_dict))
   
     separator_dict = nested_dict_to_separator_dict(cfg)

     for k, v in separator_dict.items():
          is_bool = v == type(True)
          dtype = str2bool if is_bool else v
          parser.add_argument(f'--{k}', type=dtype)

     return parser

def get_darpa_tc_node_feats_from_cfg(cfg):
    features = cfg.preprocessing.build_graphs.node_label_features
    return {
        "subject": list(map(lambda x: x.strip(), features.subject.split(","))),
        "file": list(map(lambda x: x.strip(), features.file.split(","))),
        "netflow": list(map(lambda x: x.strip(), features.netflow.split(","))),
    }

########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
     'EVENT_EXECUTE',
     'EVENT_LSEEK',
     'EVENT_MMAP',
     'EVENT_OPEN',
     'EVENT_ACCEPT',
     'EVENT_READ',
     'EVENT_RECVFROM',
     'EVENT_RECVMSG',
     'EVENT_READ_SOCKET_PARAMS',
     'EVENT_CHECK_FILE_ATTRIBUTES'
]

# The following edges are not considered to construct the
# temporal graph for experiments.
exclude_edge_type= set([
     'EVENT_FCNTL',                          # EVENT_FCNTL does not have any predicate
     'EVENT_OTHER',                          # EVENT_OTHER does not have any predicate
     'EVENT_ADD_OBJECT_ATTRIBUTE',           # This is used to add attributes to an object that was incomplete at the time of publish
     'EVENT_FLOWS_TO',                       # No corresponding system call event
])

rel2id = {
        1: 'EVENT_CONNECT',
        'EVENT_CONNECT': 1,
        2: 'EVENT_EXECUTE',
        'EVENT_EXECUTE': 2,
        3: 'EVENT_OPEN',
        'EVENT_OPEN': 3,
        4: 'EVENT_READ',
        'EVENT_READ': 4,
        5: 'EVENT_RECVFROM',
        'EVENT_RECVFROM': 5,
        6: 'EVENT_RECVMSG',
        'EVENT_RECVMSG': 6,
        7: 'EVENT_SENDMSG',
        'EVENT_SENDMSG': 7,
        8: 'EVENT_SENDTO',
        'EVENT_SENDTO': 8,
        9: 'EVENT_WRITE',
        'EVENT_WRITE': 9,
        10: 'EVENT_CLONE',
        'EVENT_CLONE': 10,
    }

ntype2id ={
     1: 'subject',
     'subject': 1,
     2: 'file',
     'file': 2,
     3: 'netflow',
     'netflow': 3,
}
