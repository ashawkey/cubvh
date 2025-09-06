import torch
from cubvh.api import cuHashTable


def _cuda_int(arr: torch.Tensor) -> torch.Tensor:
    return arr.to(dtype=torch.int32, device='cuda', non_blocking=True).contiguous()


def test_3d_basic():
    assert torch.cuda.is_available(), "CUDA is required for this test"

    # Create a small set of 3D integer coords
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 3],
            [2, 2, 2],
            [3, 4, 5],
            [10, 0, -3],
        ],
        dtype=torch.int32,
    )
    coords = _cuda_int(coords)

    ht = cuHashTable(num_dims=3)
    ht.build(coords)

    # Query some present and one absent key
    queries = torch.tensor(
        [
            [0, 0, 0],      # expect 0
            [1, 2, 3],      # expect 1
            [10, 0, -3],    # expect 4
            [9, 9, 9],      # expect -1
        ],
        dtype=torch.int32,
    )
    queries = _cuda_int(queries)

    out = ht.search(queries)
    out_cpu = out.cpu().tolist()
    assert out_cpu[0] == 0, f"expected 0, got {out_cpu[0]}"
    assert out_cpu[1] == 1, f"expected 1, got {out_cpu[1]}"
    assert out_cpu[2] == 4, f"expected 4, got {out_cpu[2]}"
    assert out_cpu[3] == -1, f"expected -1, got {out_cpu[3]}"


def test_4d_basic():
    assert torch.cuda.is_available(), "CUDA is required for this test"

    # 4D keys
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [2, 2, 2, 2],
        ],
        dtype=torch.int32,
    )
    coords = _cuda_int(coords)

    ht = cuHashTable(num_dims=4)
    ht.build(coords)

    queries = torch.tensor(
        [
            [1, 2, 3, 4],  # expect 1
            [9, 9, 9, 9],  # expect -1
        ],
        dtype=torch.int32,
    )
    queries = _cuda_int(queries)

    out = ht.search(queries).cpu().tolist()
    assert out[0] == 1, f"expected 1, got {out[0]}"
    assert out[1] == -1, f"expected -1, got {out[1]}"


def test_large_random(num_dims: int = 3, num_keys: int = 200000, num_queries: int = 100000):
    assert torch.cuda.is_available(), "CUDA is required for this test"

    # Generate random integer coords in a moderate range to create collisions
    torch.manual_seed(42)
    low, high = -5000, 5000
    coords = torch.randint(low, high, (num_keys, num_dims), dtype=torch.int32)
    coords = _cuda_int(coords)

    ht = cuHashTable(num_dims=num_dims)
    ht.build(coords)

    # Construct queries: half positives sampled from coords, half negatives random not-guaranteed present
    idx = torch.randperm(num_keys, device=coords.device)[: num_queries // 2]
    pos_queries = coords[idx]
    neg_queries = torch.randint(low, high, (num_queries - pos_queries.shape[0], num_dims), dtype=torch.int32, device=coords.device)
    queries = torch.cat([pos_queries, neg_queries], dim=0)

    out = ht.search(queries)

    # Verify that positives map to their original indices; allows any correct index if duplicates exist
    out_host = out.cpu()
    pos_idx_host = idx.cpu()
    pos_queries_host = pos_queries.cpu()
    coords_host = coords.cpu()

    # Build a dictionary from tuple(coord) to list of indices to handle duplicates
    from collections import defaultdict
    buckets = defaultdict(list)
    for i in range(coords_host.size(0)):
        buckets[tuple(coords_host[i].tolist())].append(i)

    # Check positives
    for j in range(pos_queries_host.size(0)):
        q = tuple(pos_queries_host[j].tolist())
        assert out_host[j].item() in buckets[q], f"positive query not found or wrong index at {j}: got {out_host[j].item()} for key {q}"

    # Negatives are probabilistic; just print hit rate for manual inspection
    neg_out = out_host[pos_queries_host.size(0):]
    neg_hits = (neg_out >= 0).sum().item()
    print(f"[stress] negatives that matched existing keys (expected small): {neg_hits}/{neg_out.numel()}")


def test_edge_values():
    # Test near INT32 boundaries and duplicates
    coords = torch.tensor([
        [0, 0, 0],
        [2147483647, -2147483648, 0],
        [-2147483648, 2147483647, -2147483648],
        [123, 456, 789],
        [123, 456, 789],  # duplicate
    ], dtype=torch.int32)
    coords = _cuda_int(coords)

    ht = cuHashTable(num_dims=3)
    ht.build(coords)

    queries = torch.tensor([
        [2147483647, -2147483648, 0],
        [-2147483648, 2147483647, -2147483648],
        [123, 456, 789],
        [7,7,7],
    ], dtype=torch.int32)
    queries = _cuda_int(queries)

    out = ht.search(queries).cpu().tolist()
    assert out[0] in (1,), f"unexpected index for edge case 0: {out[0]}"
    assert out[1] in (2,), f"unexpected index for edge case 1: {out[1]}"
    assert out[2] in (3,4), f"unexpected index for duplicate key: {out[2]}"
    assert out[3] == -1, f"unexpected hit for absent key: {out[3]}"


if __name__ == "__main__":
    print("Running cuHashTable tests...")
    test_3d_basic()
    test_4d_basic()
    test_large_random()
    test_edge_values()
    print("All cuHashTable tests passed.")


