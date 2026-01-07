import argparse

import d3rlpy_marin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    dataset, env = d3rlpy_marin.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy_marin.seed(args.seed)
    d3rlpy_marin.envs.seed_env(env, args.seed)

    prdc = d3rlpy_marin.algos.PRDCConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        alpha=2.5,
        update_actor_interval=2,
        observation_scaler=d3rlpy_marin.preprocessing.StandardObservationScaler(),
        compile_graph=args.compile,
    ).create(device=args.gpu)

    prdc.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy_marin.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"PRDC_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
