from collections.abc import Sequence

import torch
import torch.distributed as dist


def merge_distributed(data_list, max_len=None):
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        # rank = dist.get_rank()
    else:
        world_size = 1
        # rank = 0
    # print(f"{rank = }, {world_size = }")
    merged = []

    def gather(data: torch.Tensor) -> list[torch.Tensor]:
        # print(f"{rank = }, {data = }")
        data_size = [torch.zeros(data.dim(), dtype=torch.long, device=data.device) for _ in range(world_size)]
        # print(f"{rank = }, {data_size = }")
        dist.all_gather(data_size, torch.tensor(data.size(), device=data.device))
        # print(f"{rank = }, {data_size = }")
        data_chunks = [torch.zeros(tuple(s.cpu().numpy()), dtype=data.dtype, device=data.device) for s in data_size]
        # print(f"{rank = }, {data_chunks = }")
        # data_chunks[data.device.index] = data
        # print(f"{rank = }, {data_chunks = }")
        dist.all_gather(data_chunks, data)
        # print(f"{rank = }, broadcasting done")
        return data_chunks

    for data in data_list:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            if isinstance(data, Sequence):
                data_chunks = []
                for d in data:
                    chunks_ = gather(d)
                    data_ = torch.cat(chunks_)
                    data_chunks.append(data_)
                merged.append(data_chunks)
            else:
                _chunks = gather(data)
                merged.extend(_chunks)
        else:
            merged.append(data)

    return join_chunks(merged, max_len)


def join_chunks(chunks, max_len=None):
    if not isinstance(chunks[0], Sequence):
        merged = torch.cat([m.cpu() for m in chunks])
        if max_len is not None:
            return merged[:max_len]
        else:
            return merged
    else:
        data_list = []
        for d in zip(*chunks):
            data = torch.cat([x.cpu() for x in d])
            if max_len is not None:
                data = data[:max_len]
            data_list.append(data)
        return data_list
