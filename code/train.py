import ray
import yaml

from ray.rllib.agents.ppo.ppo import PPOTrainer

from code.model.ccppo_transformer import CCPPOTorchPolicy
from code.utils import *


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
            experiments = yaml.safe_load(f)
        create_experiment(args, experiments)
    else:
        raise Exception("No Config File")



    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)
