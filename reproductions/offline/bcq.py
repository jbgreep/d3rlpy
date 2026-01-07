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

    vae_encoder = d3rlpy_marin.models.encoders.VectorEncoderFactory([750, 750])
    rl_encoder = d3rlpy_marin.models.encoders.VectorEncoderFactory([400, 300])

    bcq = d3rlpy_marin.algos.BCQConfig(
        actor_encoder_factory=rl_encoder,
        actor_learning_rate=1e-3,
        critic_encoder_factory=rl_encoder,
        critic_learning_rate=1e-3,
        imitator_encoder_factory=vae_encoder,
        imitator_learning_rate=1e-3,
        batch_size=100,
        lam=0.75,
        action_flexibility=0.05,
        n_action_samples=100,
        compile_graph=args.compile,
    ).create(args.gpu)

    bcq.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy_marin.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"BCQ_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
