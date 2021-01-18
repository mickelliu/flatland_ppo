import argparse
import ray
from ray.tune.config_parser import make_parser
from ray.cluster_utils import Cluster
from ray.rllib.utils import merge_dicts
from ray.tune.logger import TBXLogger

from code.wandblogger import WandbLogger

from envs.flatland import get_eval_config

def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog="")

    parser.add_argument("-f",
                        "--config-file",
                        default=None,
                        type=str,
                        help="If specified, use config options from this file. \
                            Note that this overrides any trial-specific options set via flags above.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--bind-all",
        action="store_true",
        default=False,
        help="Whether to expose on network (binding on all network interfaces).")
    parser.add_argument(
        "--log-flatland-stats",
        action="store_true",
        default=True,
        help="Whether to log additional flatland specfic metrics such as percentage complete or normalized score.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Whether to run evaluation. Default evaluation config is default.yaml "
        "to use custom evaluation config set (eval_generator:test_eval) under configs")
    parser.add_argument(
        "--torch",
        action="store_true")

    return parser


def on_episode_end(info):
    episode = info["episode"]  # type: MultiAgentEpisode

    episode_steps = 0
    episode_max_steps = 0
    episode_num_agents = 0
    episode_score = 0
    episode_done_agents = 0
    episode_num_swaps = 0

    for agent, agent_info in episode._agent_to_last_info.items():
        if episode_max_steps == 0:
            episode_max_steps = agent_info["max_episode_steps"]
            episode_num_agents = agent_info["num_agents"]
        episode_steps = max(episode_steps, agent_info["agent_step"])
        episode_score += agent_info["agent_score"]
        if "num_swaps" in agent_info:
            episode_num_swaps += agent_info["num_swaps"]
        if agent_info["agent_done"]:
            episode_done_agents += 1

    # Not a valid check when considering a single policy for multiple agents
    #assert len(episode._agent_to_last_info) == episode_num_agents

    norm_factor = 1.0 / (episode_max_steps * episode_num_agents)
    percentage_complete = float(episode_done_agents) / episode_num_agents

    episode.custom_metrics["episode_steps"] = episode_steps
    episode.custom_metrics["episode_max_steps"] = episode_max_steps
    episode.custom_metrics["episode_num_agents"] = episode_num_agents
    episode.custom_metrics["episode_return"] = episode.total_reward
    episode.custom_metrics["episode_score"] = episode_score
    episode.custom_metrics["episode_score_normalized"] = episode_score * norm_factor
    episode.custom_metrics["episode_num_swaps"] = episode_num_swaps / 2
    episode.custom_metrics["percentage_complete"] = percentage_complete

def create_experiment(args, experiments):

    verbose = 1
    custom_fn = False
    webui_host = "localhost"
    for exp in experiments.values():

        if args.torch:
            exp["config"]["use_pytorch"] = True
        if args.v:
            exp["config"]["log_level"] = "INFO"
            verbose = 2
        if args.vv:
            exp["config"]["log_level"] = "DEBUG"
            verbose = 3
        if args.log_flatland_stats:
            exp['config']['callbacks'] = {'on_episode_end': on_episode_end}

        if args.eval:
            eval_configs_file = exp['config'].get('env_config', {}).get('eval_generator', "default")

            if args.record:
                eval_configs_file = exp['config'].get('env_config', {}).get('eval_generator', "default_render")

            eval_configs = get_eval_config(eval_configs_file)
            eval_seed = eval_configs.get('evaluation_config', {}).get('env_config', {}).get('seed')
            eval_render = eval_configs.get('evaluation_config', {}).get('env_config', {}).get('render')

            # add evaluation config to the current config
            exp['config'] = merge_dicts(exp['config'], eval_configs)
            if exp['config'].get('evaluation_config'):
                exp['config']['evaluation_config']['env_config'] = exp['config'].get('env_config')
                eval_env_config = exp['config']['evaluation_config'].get('env_config')
                if eval_seed and eval_env_config:
                    # We override the env seed from the evaluation config
                    eval_env_config['seed'] = eval_seed
                if eval_render and eval_env_config:
                    # We override the env render from the evaluation config
                    eval_env_config['render'] = eval_render
                    # Set video_dir if it exists
                    eval_render_dir = eval_configs.get('evaluation_config', {}).get('env_config', {}).get('video_dir')
                    if eval_render_dir:
                        eval_env_config['video_dir'] = eval_render_dir
                # Remove any wandb related configs
                if eval_env_config:
                    if eval_env_config.get('wandb'):
                        del eval_env_config['wandb']

            # Remove any wandb related configs
            if exp['config']['evaluation_config'].get('wandb'):
                del exp['config']['evaluation_config']['wandb']
        if args.save_checkpoint:
            exp['config']['env_config']['save_checkpoint'] = True
        if args.config_file:
            exp['config']['env_config']['yaml_config'] = args.config_file
        exp['loggers'] = [WandbLogger, TBXLogger]

        # global checkpoint_freq, keep_checkpoints_num, checkpoint_score_attr, checkpoint_at_end
        #
        # checkpoint_freq = exp['checkpoint_freq']
        # keep_checkpoints_num = exp['keep_checkpoints_num']
        # checkpoint_score_attr = exp['checkpoint_score_attr']
        # checkpoint_at_end = exp['checkpoint_at_end']

        ray.init(
            num_gpus=args.ray_num_gpus,
            webui_host=webui_host)