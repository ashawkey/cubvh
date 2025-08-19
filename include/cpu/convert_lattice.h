#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <tuple>

/* This file contains CPU functions for converting between:
1. Lattice/Corner representation (abcdefghi): corner coordinates [N, 3], each corner contains a single float value [N, 1]
2. Cell/Voxel representation (ABCD): voxel coordinates [N, 3], each voxel contains 2^D corner values [N, 2^D].

a ------ b ------ c
|        |        |
|   A    |   B    |
|        |        |
d ------ e ------ f
|        |        |
|   C    |   D    |
|        |        |
g ------ h ------ i

*/


namespace cubvh {
namespace cpu {

// corner offsets (standard marching cubes order) are defined locally in functions to avoid ODR conflicts.

inline std::tuple<int*, float*, int>
voxels2corners(const int* coords, const float* corners, int N) {
// input voxels: coords [N*3], corners [N*8], count N
// output corners: coords [M*3], values [M], count M
// corner values are averaged if multiple voxels cover the same corner.
	// Accumulate sums and counts per unique corner coordinate
	struct Key3 { int x,y,z; bool operator==(const Key3& o) const noexcept { return x==o.x && y==o.y && z==o.z; } };
	struct Key3Hash { size_t operator()(Key3 const& k) const noexcept {
		// simple hash combine
		size_t h = std::hash<int>{}(k.x);
		h ^= std::hash<int>{}(k.y) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
		h ^= std::hash<int>{}(k.z) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
		return h;
	}};

	std::unordered_map<Key3, std::pair<double,int>, Key3Hash> acc; // sum, cnt
	acc.reserve(static_cast<size_t>(N) * 8);

	// local corner offsets
	const int off[8][3] = {
		{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
		{0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
	};

	for (int i = 0; i < N; ++i) {
		const int bx = coords[3*i+0];
		const int by = coords[3*i+1];
		const int bz = coords[3*i+2];
		for (int k = 0; k < 8; ++k) {
			Key3 key{bx + off[k][0], by + off[k][1], bz + off[k][2]};
			auto it = acc.find(key);
			const float v = corners[8*i + k];
			if (it == acc.end()) {
				acc.emplace(key, std::make_pair(static_cast<double>(v), 1));
			} else {
				it->second.first += static_cast<double>(v);
				it->second.second += 1;
			}
		}
	}

	// Move to vector and sort for deterministic order
	std::vector<std::pair<Key3, float>> items;
	items.reserve(acc.size());
	for (const auto& kv : acc) {
		const Key3& k = kv.first;
		const double sum = kv.second.first;
		const int cnt = kv.second.second;
		items.emplace_back(k, static_cast<float>(sum / std::max(1, cnt)));
	}
	std::sort(items.begin(), items.end(), [](const auto& a, const auto& b){
		if (a.first.x != b.first.x) return a.first.x < b.first.x;
		if (a.first.y != b.first.y) return a.first.y < b.first.y;
		return a.first.z < b.first.z;
	});

	const int M = static_cast<int>(items.size());
	int* outCoords = (M > 0) ? new int[M * 3] : nullptr;
	float* outValues = (M > 0) ? new float[M] : nullptr;
	for (int i = 0; i < M; ++i) {
		outCoords[3*i+0] = items[i].first.x;
		outCoords[3*i+1] = items[i].first.y;
		outCoords[3*i+2] = items[i].first.z;
		outValues[i] = items[i].second;
	}

	return std::make_tuple(outCoords, outValues, M);
}

inline std::tuple<int*, float*, int>
corners2voxels(const int* coords, const float* values, int N) {
// input corners: coords [M*3], values [M], count M
// output voxels: coords [N*3], corners [N*8], count N
// negative values mean inside, if a corner is negative and it has undefined neigbor corners, these neighbor corners will be default to 0 when converting to voxels.
	// Map existing corners
	struct Key3 { int x,y,z; bool operator==(const Key3& o) const noexcept { return x==o.x && y==o.y && z==o.z; } };
	struct Key3Hash { size_t operator()(Key3 const& k) const noexcept {
		size_t h = std::hash<int>{}(k.x);
		h ^= std::hash<int>{}(k.y) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
		h ^= std::hash<int>{}(k.z) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
		return h;
	}};

	std::unordered_map<Key3, float, Key3Hash> cornerMap;
	cornerMap.reserve(static_cast<size_t>(N));
	for (int i = 0; i < N; ++i) {
		Key3 k{coords[3*i+0], coords[3*i+1], coords[3*i+2]};
		cornerMap.emplace(k, values[i]);
	}

	// Accumulate voxel candidates
	struct VoxelAccum { float cv[8]; bool has[8]; VoxelAccum(){ for(int i=0;i<8;++i){ cv[i]=0.0f; has[i]=false; } } };
	std::unordered_map<Key3, VoxelAccum, Key3Hash> vox;
	vox.reserve(static_cast<size_t>(N) * 8);

	// local corner offsets
	const int off[8][3] = {
		{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
		{0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
	};

	for (int i = 0; i < N; ++i) {
		const int cx = coords[3*i+0];
		const int cy = coords[3*i+1];
		const int cz = coords[3*i+2];
		const float v = values[i];
		// this corner contributes to 8 possible voxels with base = corner - offset[k]
		for (int k = 0; k < 8; ++k) {
			Key3 base{cx - off[k][0], cy - off[k][1], cz - off[k][2]};
			VoxelAccum &rec = vox[base];
			rec.cv[k] = v;
			rec.has[k] = true;
		}
	}

	// Collect valid voxels: include if (all 8 defined) or (any defined corner negative)
	std::vector<std::pair<Key3, VoxelAccum>> vlist;
	vlist.reserve(vox.size());
	for (auto &kv : vox) {
		const VoxelAccum &rec = kv.second;
		bool anyDefined = false;
		bool anyNeg = false;
		bool allDefined = true;
		for (int k = 0; k < 8; ++k) {
			anyDefined = anyDefined || rec.has[k];
			allDefined = allDefined && rec.has[k];
			if (rec.has[k] && rec.cv[k] < 0.0f) anyNeg = true;
		}
		if (!anyDefined) continue; // shouldn't happen
		if (allDefined || anyNeg) {
			vlist.emplace_back(kv.first, kv.second);
		}
	}

	// Sort for deterministic order
	std::sort(vlist.begin(), vlist.end(), [](const auto& a, const auto& b){
		if (a.first.x != b.first.x) return a.first.x < b.first.x;
		if (a.first.y != b.first.y) return a.first.y < b.first.y;
		return a.first.z < b.first.z;
	});

	const int M = static_cast<int>(vlist.size());
	int* outCoords = (M > 0) ? new int[M * 3] : nullptr;
	float* outCorners = (M > 0) ? new float[M * 8] : nullptr;
	for (int i = 0; i < M; ++i) {
		const Key3 &b = vlist[i].first;
		const VoxelAccum &rec = vlist[i].second;
		bool anyNeg = false;
		for (int k = 0; k < 8; ++k) if (rec.has[k] && rec.cv[k] < 0.0f) { anyNeg = true; break; }
		outCoords[3*i+0] = b.x;
		outCoords[3*i+1] = b.y;
		outCoords[3*i+2] = b.z;
		for (int k = 0; k < 8; ++k) {
			float v = rec.has[k] ? rec.cv[k] : (anyNeg ? 0.0f : 0.0f);
			// The rule says we default undefined neighbors to 0 only if there is a negative corner.
			// If allDefined, no missing; if not anyNeg, this voxel wasn't included above.
			outCorners[8*i + k] = v;
		}
	}

	return std::make_tuple(outCoords, outCorners, M);
}

} // namespace cpu
} // namespace cubvh
