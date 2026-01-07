import argparse

import d3rlpy_marin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    dataset, env = d3rlpy_marin.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy_marin.seed(args.seed)
    d3rlpy_marin.envs.seed_env(env, args.seed)

    sac = d3rlpy_marin.algos.SACConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256,
        compile_graph=args.compile,
    ).create(device=args.gpu)

    sac.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy_marin.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"SAC_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
