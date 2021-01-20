import ray
import yaml

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune import run_experiments

from _achived.ccppo_concatenate import CCPPOTorchPolicy
# from ccppo_transformer import CCPPOTorchPolicy
from _achived.utils import create_parser, create_experiment

parser = create_parser()


def get_policy_class(config):
    if config["framework"] == "torch":
        return CCPPOTorchPolicy


CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOTorchPolicy,
    get_policy_class=get_policy_class,
)

if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file) as f:
            specs = yaml.safe_load(f)
            print(specs)
        exp = create_experiment(args, specs)
    else:
        raise Exception("No Config File")

    run_experiments(exp, verbose=1)

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)
