import submitit, os, sys, argparse
osp = os.path

from cnerf import train
from cnerf import args as cargs

class Trainer(object):
    def __init__(self, args):
        self.args = args
        
    def __call__(self):
        job_env = submitit.JobEnvironment()
        self.args.logs_path = osp.join(self.args.logs_path, self.args.name)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        train.main(self.args)

    def checkpoint(self):
        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = osp.join(self.args.logs_path, "checkpoint.pth")
        if osp.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)


def slurm_args(parser):
    parser.add_argument("--ngpus", default=4, type=int)
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--timeout", default=9000, type=int)
    parser.add_argument("--partition", default="QRTX5000", type=str)
    return parser

def main():
    args = cargs.parse_args(slurm_args)
    
    executor = submitit.AutoExecutor(folder=args.logs_path, slurm_max_num_timeout=30)
    executor.update_parameters(
        name=args.name,
        mem_gb=40 * args.ngpus,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,
        cpus_per_task=8, #10
        nodes=args.nodes,
        timeout_min=args.timeout,
        slurm_partition=args.partition,
        slurm_exclude="bergamot,perilla,caraway,cassia",
    )
    
    trainer = Trainer(args)
    job = executor.submit(trainer)
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()