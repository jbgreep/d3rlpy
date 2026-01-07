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

    if "halfcheetah" in args.dataset:
        kernel = "gaussian"
    else:
        kernel = "laplacian"

    bear = d3rlpy_marin.algos.BEARConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        imitator_learning_rate=3e-4,
        alpha_learning_rate=1e-3,
        imitator_encoder_factory=vae_encoder,
        temp_learning_rate=0.0,
        initial_temperature=1e-20,
        batch_size=256,
        mmd_sigma=20.0,
        mmd_kernel=kernel,
        n_mmd_action_samples=4,
        alpha_threshold=0.05,
        n_target_samples=10,
        n_action_samples=100,
        warmup_steps=40000,
        compile_graph=args.compile,
    ).create(device=args.gpu)

    bear.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy_marin.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"BEAR_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
