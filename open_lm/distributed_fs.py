import glob
import logging
import os
import pickle
import re
import subprocess
import sys
import random
from datetime import datetime
import functools
import numpy as np
from functools import partial
from pathlib import Path
import json
import webdataset as wds
import io
import braceexpand
from sklearn.linear_model import LogisticRegression
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample



import torch
from torch import optim
from torch.cuda.amp import GradScaler
from .precision import get_autocast

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .model import Block
from .losses import CrossEntropyLossWithZLoss

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from open_lm.model import create_model
from .data import get_data, get_wds_dataset
from .distributed import is_master, init_distributed_device, broadcast_object
from .logger import setup_logging
from .params import parse_args
from .scheduler import cosine_lr
from .train import train_one_epoch, evaluate
from .file_utils import (
    pt_load,
    check_exists,
    start_sync_process,
    remote_sync,
    get_string_for_epoch,
)


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(" ")[-1])
            for x in result.stdout.decode().split("\n")[:-1]
        ]
    else:
        checkpoints = glob.glob(path + "**/epoch_*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def get_state_dict(name):
    checkpoint = pt_load(name, map_location="cpu")
    if "epoch" in checkpoint:
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
    else:
        sd = checkpoint
    return sd


def load_model(args, model):
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
        logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    return start_epoch


def load_optimizer(args, model, optimizer, scaler):
    potential_checkpoint = args.resume.replace("epoch_", "optimizer_")
    if check_exists(potential_checkpoint):
        checkpoint = pt_load(potential_checkpoint, map_location="cpu")
    else:
        checkpoint = pt_load(args.resume, map_location="cpu")
    if "optimizer" in checkpoint:
        if optimizer is not None:
            osd = checkpoint["optimizer"]
            if args.fsdp:
                osd = FSDP.optim_state_dict_to_load(osd, model, optimizer)
            optimizer.load_state_dict(osd)
            logging.info(f"=> resuming optimizer")
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        logging.info(f"=> WARNING: not resuming optimizer.")


def save_checkpoint(args, model, optimizer, scaler, completed_epoch, evaluation_loss):
    cpu_state, optim_state = None, None
    if args.logs and args.logs.lower() != "none" and args.fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)

    if args.save_logs:
        checkpoint_dict_model = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": cpu_state if args.fsdp else model.state_dict(),
            "evaluation_loss": evaluation_loss,
        }
        checkpoint_dict_opt = {
            "epoch": completed_epoch,
            "name": args.name,
            "optimizer": optim_state if args.fsdp else optimizer.state_dict(),
            "evaluation_loss": evaluation_loss,
        }
        if scaler is not None:
            checkpoint_dict_opt["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict_model,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
            torch.save(
                checkpoint_dict_opt,
                os.path.join(args.checkpoint_path, f"optimizer_{completed_epoch}.pt"),
            )

        if args.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(
                args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
            )
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)
            previous_checkpoint = os.path.join(
                args.checkpoint_path, f"optimizer_{completed_epoch - 1}.pt"
            )
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
            ]
        )

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
        )
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print(
                    "Error. Cannot use save-most-recent with remote_sync and resume latest."
                )
                return -1
            if args.remote_sync_protocol != "s3":
                print("Error. Sync protocol not supported when using resume latest.")
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(
                    checkpoint_path, remote=args.remote_sync is not None
                )
            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("remote sync successful.")
        else:
            logging.info("Error: remote sync failed. Exiting.")
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        remote_sync_process.start()

    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )

    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    random_seed(args.seed, 0)
    model = create_model(args)
    args.vocab_size = model.vocab_size
    args.seq_len = model.seq_len
    model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # optionally resume model from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_model(args, model)
    elif args.pretrained is not None:
        print("=> loading from a pre-trained model.")
        args.resume = args.pretrained
        ep = load_model(args, model)
        # this flag continues training from the pre-trained model.
        if args.load_pretrained_state:
            start_epoch = ep
        else:
            args.resume = None
    elif args.average is not None:
        num_models_to_average = len(args.average)
        print(
            "=> Averaging models: ",
            args.average,
            " with coefficients: ",
            args.average_coefficients,
        )
        assert (
            num_models_to_average > 1
        ), "num_models_to_average must be > 1 - else use --pretrained"
        if args.average_coefficients is None:
            args.average_coefficients = [
                1.0 / num_models_to_average
            ] * num_models_to_average
        else:
            assert len(args.average_coefficients) == num_models_to_average
        state_dict = {
            k: v * args.average_coefficients[0]
            for k, v in get_state_dict(args.average[0]).items()
        }
        for i in range(1, num_models_to_average):
            state_dict_i = get_state_dict(args.average[i])
            for k in state_dict:
                state_dict[k] = (
                    state_dict[k] + state_dict_i[k] * args.average_coefficients[i]
                )
        model.load_state_dict(state_dict)

    if args.distributed:
        if args.fsdp:
            # from https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
            transformer_auto_wrapper_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    Block,
                },
            )
            # tries to follow gopher...
            mp_policy = None
            if args.fsdp_amp:
                print("=> using bfloat16 params as part of fsdp amp policy.")
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )

            if args.rank == 0:
                print(
                    f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())}"
                )
                print(f"Before FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB")

            fsdp_kwargs = {}
            assert not (args.fsdp_hybrid and args.fsdp_hybrid_o2), "Only --fsdp-hybrid or --fsdp-hybrid-o2 should be set."
            if args.fsdp_backward_prefetch:
                fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            if args.fsdp_hybrid:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
            if args.fsdp_hybrid_o2:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy._HYBRID_SHARD_ZERO2
            print("=> FSDP kwargs: ", fsdp_kwargs)

            # init FSDP
            model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrapper_policy,
                device_id=device,
                mixed_precision=mp_policy,
                cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
                use_orig_params=args.fsdp_use_orig_params,
                limit_all_gathers=args.fsdp_limit_all_gathers,
                **fsdp_kwargs,
            )

            print(
                f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
            )
            print(
                f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
            )
        else:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args["static_graph"] = True
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], **ddp_args
            )



    # Function to preprocess each numpy array
    def prep(tensor):
        appended_tensor = tensor

        # Append 1024 to each row and add to the batch list
        to_append = torch.full((tensor.shape[0], 1), 1024)
        batch_tensor = torch.cat((appended_tensor, to_append), dim=1)

        return batch_tensor

    def preprocess(sample, reps):
        # Assert input shapes
        assert sample.shape == (1, 256)
        assert reps.shape[0] == 10 and reps.shape[2] == 256  # We don't assert the middle dim to allow variability

        N = reps.shape[1]

        # Step 1: Append the sample to the reps making the shape (10, N+1, 256)
        appended_sample = np.repeat(sample, 10, axis=0).reshape(10, 1, 256)
        updated_reps = np.concatenate([reps, appended_sample], axis=1)

        # Step 2: Make the shape (10, N+1, 257) by appending 1024 to the last dimension
        with_separator = np.concatenate([updated_reps, np.full((10, N+1, 1), 1024)], axis=-1)

        # Step 3: Flatten the last two dimensions
        flattened = with_separator.reshape(10, -1)

        # Step 4: Return
        return torch.from_numpy(flattened)

    a, l = 0, 0
    autocast = get_autocast(args.precision)

    # Function to decode numpy arrays from binary streams
    def numpy_decoder(data):
        array = np.load(io.BytesIO(data))
        return array
    def log_and_continue(exn):
        """Call in an exception handler to ignore any exception, issue a warning, and continue."""
        logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
        return True
    def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
        """Return function over iterator that groups key, value pairs into samples.

        :param keys: function that splits the key into key and extension (base_plus_ext)
        :param lcase: convert suffixes to lower case (Default value = True)
        """
        current_sample = None
        for filesample in data:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            if prefix is None:
                continue
            if lcase:
                suffix = suffix.lower()
            # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
            #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
            #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
            if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        if valid_sample(current_sample):
            yield current_sample

    def tarfile_to_samples_nothrow(src, handler=log_and_continue):
        # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
        streams = url_opener(src, handler=handler)
        files = tar_file_expander(streams, handler=handler)
        samples = group_by_keys_nothrow(files, handler=handler)
        return samples

    # Partition the WebDataset across workers
    urls = list(braceexpand.braceexpand(args.train_data[0]))
    pipeline = [wds.SimpleShardList(urls)]
    pipeline.extend([
        wds.split_by_node,
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
    ])
    dataset = wds.DataPipeline(*pipeline)
    loss = torch.nn.CrossEntropyLoss(reduction="none")

    reps = np.load(args.train_data_upsampling_factors)

    cor_all = []

    N_EXAMPLES = args.train_num_samples
    N_EXAMPLES_WORKER = N_EXAMPLES // args.world_size

    # Iterate over the dataset
    with torch.no_grad(), autocast():
        for sample in dataset:

            np_arr = numpy_decoder(sample["npy"])
            cls = int(sample["txt"].decode('utf-8'))

            options = preprocess(np_arr, reps).to(args.device)
            batch_x = options[:, :-1]
            batch_y = options[:, 1:]

            losses = []
            for i in range(10):
                out, _ = model(batch_x[i:i+1])
                loss_value = loss(out.reshape(-1, args.vocab_size), batch_y[i:i+1].reshape(-1))
                loss_value = loss_value[-257:].mean()

                losses.append(loss_value.item())

            losses = torch.tensor(losses)
            pred = losses.argmin().item()

            cor = (pred == cls)
            cor_all.append(cor)

            if args.rank == 0:
                print(len(cor_all))

            if len(cor_all) == N_EXAMPLES_WORKER:
                break

    dist.barrier()

    cor_avg = torch.tensor([int(c) for c in cor_all]).float().mean().to(args.device)
    # cor_tensor = torch.cat(cor_all).uint8().to(args.device)

    dist.all_reduce(cor_avg, op=dist.ReduceOp.SUM)
    cor_avg /= args.world_size

    if args.rank == 0:
        print(args.resume)
        print(cor_avg)

    '''
    # Create storage lists for gathered data, making sure they're on the same CUDA device
    cor_shape = cor_tensor.shape

    cor_list = [torch.zeros(cor_shape, device=args.device) for _ in range(args.world_size)]

    # Use torch.distributed.all_gather to collect embeddings and labels from all processes
    dist.barrier()
    dist.all_gather(cor_list, cor_tensor)
    dist.barrier()

    if args.rank == 0:
        print(cor_list.mean())
    '''

if __name__ == "__main__":
    main(sys.argv[1:])
