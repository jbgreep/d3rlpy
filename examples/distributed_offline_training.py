import d3rlpy_marin

# This script needs to be launched by using torchrun command.
# $ torchrun \
#   --nnodes=1 \
#   --nproc_per_node=3 \
#   --rdzv_id=100 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29400 \
#   examples/distributed_offline_training.py


def main() -> None:
    # GPU version:
    # rank = d3rlpy.distributed.init_process_group("nccl")
    rank = d3rlpy_marin.distributed.init_process_group("gloo")
    print(f"Start running on rank={rank}.")

    # GPU version:
    # device = f"cuda:{rank}"
    device = "cpu:0"

    # setup algorithm
    cql = d3rlpy_marin.algos.CQLConfig(
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        alpha_learning_rate=1e-3,
    ).create(device=device, enable_ddp=True)

    # prepare dataset
    dataset, env = d3rlpy_marin.datasets.get_pendulum()

    # disable logging on rank != 0 workers
    logger_adapter: d3rlpy_marin.logging.LoggerAdapterFactory
    evaluators: dict[str, d3rlpy_marin.metrics.EvaluatorProtocol]
    if rank == 0:
        evaluators = {"environment": d3rlpy_marin.metrics.EnvironmentEvaluator(env)}
        logger_adapter = d3rlpy_marin.logging.FileAdapterFactory()
    else:
        evaluators = {}
        logger_adapter = d3rlpy_marin.logging.NoopAdapterFactory()

    # start training
    cql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        evaluators=evaluators,
        logger_adapter=logger_adapter,
        show_progress=rank == 0,
    )

    d3rlpy_marin.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
