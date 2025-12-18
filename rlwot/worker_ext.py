import gc
import time
import torch
import vllm.distributed.parallel_state as ps

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    """

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        print(f"perturb_self_weights tp_rank={ps.get_tp_group().rank} tp_size={ps.get_tp_group().world_size}")
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        for _, p in self.model_runner.model.named_parameters():
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, SIGMA):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        for _, p in self.model_runner.model.named_parameters():
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(SIGMA) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        print(f"init_inter_engine_group tp_rank={ps.get_tp_group().rank} tp_size={ps.get_tp_group().world_size}")
        time.sleep(10)
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True