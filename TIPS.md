## Debugging with DDP

```python
import torch.distributed as dist
dist.barrier()
if args.rank == 0:
    import ipdb; ipdb.set_trace()
dist.barrier()
```
