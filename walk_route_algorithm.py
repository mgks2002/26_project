"""
반려동물 산책 경로 추천 알고리즘
=====================================
- 구글맵 JSON 데이터를 파싱해 가중치 그래프를 구성합니다.
- 도로 유형 / 주변 POI에 따라 간선 비용에 가중치를 적용합니다.
- 다익스트라 실행 전 노이즈를 간선별로 1회만 적용해 경로 다양성을 확보합니다.
- 목표 산책 거리(= 속도 × 시간)에 맞는 순환 경로를 반환합니다.

사용법:
    route = find_walk_route("map_data.json", start_node="N00", walk_minutes=30)
    print(route)

Android 연동:
    10초마다 GPS 업데이트 → 현재 노드를 찾아 start_node 갱신 후 재호출
"""

import json
import heapq
import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 1. 가중치 테이블
# ─────────────────────────────────────────────

# 도로 유형별 기본 비용 승수 (낮을수록 선호)
ROAD_TYPE_WEIGHT: Dict[str, float] = {
    "boulevard":    0.55,   # 대로 — 넓은 인도, 안전, 쾌적
    "park":         0.45,   # 공원 산책로 — 가장 선호
    "normal_road":  1.00,   # 기본 비용 기준
    "alley":        1.60,   # 골목 — 인도 없거나 좁음
    "busy_road":    1.80,   # 차량 많은 도로
    "construction": 3.00,   # 공사 현장 — 강하게 회피
}

# 근처 POI에 따른 추가 보너스 (음수 = 비용 감소 = 선호)
POI_BONUS: Dict[str, float] = {
    "pet_store":        -0.20,   # 반려동물 용품점
    "park":             -0.30,   # 공원
    "cafe":             -0.10,   # 카페 (반려동물 동반 가능)
    "fountain":         -0.10,   # 분수 / 쉼터
    "bench":            -0.10,   # 벤치
    "bus_stop":         +0.00,   # 중립
    "pharmacy":         +0.00,
    "convenience_store":+0.00,
    "construction_site":+1.50,   # 공사 현장 POI가 근처에 있으면 추가 패널티
}

# 기본 보행 속도 (반려동물 동반 시 성인 속도보다 느림)
WALK_SPEED_M_PER_MIN: float = 60.0   # ≈ 3.6 km/h

# 돌아오는 경로에서 이미 지나온 간선에 부여할 패널티 배수
# (값이 클수록 같은 길로 돌아올 가능성 감소)
REVISIT_PENALTY: float = 4.0


# ─────────────────────────────────────────────
# 2. 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class Node:
    id: str
    lat: float
    lng: float
    name: str
    type: str


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    distance_m: float
    road_type: str
    name: str
    nearby_pois: List[str]
    notes: str
    # 계산된 최종 가중치 비용 (load 시 채워짐)
    base_cost: float = 0.0


@dataclass
class RouteResult:
    node_ids: List[str]          # 경로 노드 순서
    node_names: List[str]        # 노드 이름 (표시용)
    edge_ids: List[str]          # 사용된 간선 ID
    total_distance_m: float      # 총 거리 (m)
    estimated_time_min: float    # 예상 소요 시간 (분)
    total_cost: float            # 알고리즘 내부 비용 (낮을수록 좋음)
    highlights: List[str]        # 경로 상 주요 포인트
    warnings: List[str]          # 경로 주의사항 (위험 수준만)
    notes: List[str]             # 경로 참고사항 (minor 안내)


# ─────────────────────────────────────────────
# 3. 맵 파서
# ─────────────────────────────────────────────

class MapGraph:
    """JSON 파일을 읽어 가중치 무방향 그래프를 구성합니다."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        # adjacency: node_id → [(neighbor_id, Edge)]
        self.adj: Dict[str, List[Tuple[str, Edge]]] = {}
        # 간선 ID → Edge (load 시점에 한 번만 구성)
        self.edge_map: Dict[str, Edge] = {}

    def load_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for n in data["nodes"]:
            node = Node(
                id=n["id"], lat=n["lat"], lng=n["lng"],
                name=n["name"], type=n["type"]
            )
            self.nodes[node.id] = node
            self.adj[node.id] = []

        for e in data["edges"]:
            edge = Edge(
                id=e["id"],
                from_node=e["from"],
                to_node=e["to"],
                distance_m=e["distance_m"],
                road_type=e["road_type"],
                name=e["name"],
                nearby_pois=e.get("nearby_pois", []),
                notes=e.get("notes", ""),
            )
            edge.base_cost = self._compute_base_cost(edge)

            # 무방향 그래프 (양방향 추가)
            self.adj[edge.from_node].append((edge.to_node, edge))
            self.adj[edge.to_node].append((edge.from_node, edge))

            # [수정 5] edge_map을 load 시점에 한 번만 구성 (중복 순회 제거)
            self.edge_map[edge.id] = edge

    def _compute_base_cost(self, edge: Edge) -> float:
        """
        간선의 기본 비용 = 거리 × 도로 유형 가중치 + POI 보너스 합산
        """
        road_w = ROAD_TYPE_WEIGHT.get(edge.road_type, 1.0)
        cost = edge.distance_m * road_w

        poi_bonus = sum(POI_BONUS.get(poi, 0.0) for poi in edge.nearby_pois)
        # 보너스는 거리에 비례해 적용 (최대 50% 감소 보장)
        cost = cost * (1.0 + max(poi_bonus, -0.5))

        return max(cost, 1.0)   # 최솟값 보장

    def compute_avg_cost_per_m(self) -> float:
        """
        [수정 2] 그래프 전체 간선의 실제 평균 비용/m 을 계산합니다.
        target_half_cost 산정 시 단순 1.0 대신 이 값을 사용해
        비용 단위 불일치를 해소합니다.
        """
        total_cost = 0.0
        total_dist = 0.0
        for edge in self.edge_map.values():
            total_cost += edge.base_cost
            total_dist += edge.distance_m
        if total_dist == 0:
            return 1.0
        return total_cost / total_dist


# ─────────────────────────────────────────────
# 4. 노이즈 적용 (다익스트라 실행 전 1회)
# ─────────────────────────────────────────────

def build_noisy_costs(graph: MapGraph) -> Dict[str, float]:
    """
    [수정 1] 각 간선에 노이즈를 다익스트라 실행 전 1회만 적용합니다.
    - 기존: 다익스트라 내부에서 간선을 방문할 때마다 노이즈 적용
      → 같은 간선도 방문 시점마다 비용이 달라져 최적성 보장 붕괴
    - 수정: 탐색 전에 noisy_cost 딕셔너리를 미리 생성
      → 탐색 중 비용은 고정, 매 요청마다 다른 경로 생성이라는 의도는 유지

    noise_factor ∈ [1.10, 1.20]  (균등 분포)
    """
    noisy_costs: Dict[str, float] = {}
    for edge_id, edge in graph.edge_map.items():
        noise_factor = random.uniform(1.10, 1.20)
        noisy_costs[edge_id] = edge.base_cost * noise_factor
    return noisy_costs


# ─────────────────────────────────────────────
# 5. 순환 경로 탐색 알고리즘
# ─────────────────────────────────────────────

class WalkRouteFinderError(Exception):
    pass


def _dijkstra(graph: MapGraph,
              start: str,
              edge_costs: Dict[str, float]
              ) -> Tuple[Dict[str, float], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    [수정 1] 변형 다익스트라: 외부에서 주입된 edge_costs를 사용합니다.
    노이즈는 이미 build_noisy_costs() 또는 _build_return_costs()에서
    적용된 상태이므로 내부에서 추가 적용하지 않습니다.

    Returns:
        dist    : 출발지 → 각 노드 누적 비용
        prev    : 이전 노드 (경로 역추적용)
        via_edge: 사용된 간선 ID
    """
    dist: Dict[str, float] = {nid: float("inf") for nid in graph.nodes}
    prev: Dict[str, Optional[str]] = {nid: None for nid in graph.nodes}
    via_edge: Dict[str, Optional[str]] = {nid: None for nid in graph.nodes}
    dist[start] = 0.0

    pq = [(0.0, start)]   # (누적 비용, 노드ID)

    while pq:
        cur_cost, u = heapq.heappop(pq)
        if cur_cost > dist[u]:
            continue
        for v, edge in graph.adj[u]:
            cost = edge_costs[edge.id]
            new_cost = dist[u] + cost
            if new_cost < dist[v]:
                dist[v] = new_cost
                prev[v] = u
                via_edge[v] = edge.id
                heapq.heappush(pq, (new_cost, v))

    return dist, prev, via_edge


def _build_return_costs(noisy_costs: Dict[str, float],
                        used_edge_ids: List[str]) -> Dict[str, float]:
    """
    [수정 3] 돌아오는 경로용 비용 딕셔너리를 생성합니다.
    갈 때 사용한 간선에 REVISIT_PENALTY 배수를 적용해
    같은 길로 되돌아오는 경로를 강하게 억제합니다.
    """
    return_costs = dict(noisy_costs)
    for eid in used_edge_ids:
        if eid in return_costs:
            return_costs[eid] *= REVISIT_PENALTY
    return return_costs


def _reconstruct_path(prev: Dict[str, Optional[str]],
                      via_edge: Dict[str, Optional[str]],
                      start: str, end: str
                      ) -> Tuple[List[str], List[str]]:
    """prev 딕셔너리로 경로 역추적 → (노드 리스트, 간선 리스트)"""
    nodes_path: List[str] = []
    edges_path: List[str] = []
    cur = end
    while cur != start:
        nodes_path.append(cur)
        if via_edge[cur]:
            edges_path.append(via_edge[cur])
        cur = prev[cur]
        if cur is None:
            raise WalkRouteFinderError("경로 역추적 실패: 연결되지 않은 그래프")
    nodes_path.append(start)
    nodes_path.reverse()
    edges_path.reverse()
    return nodes_path, edges_path


def _select_waypoint(graph: MapGraph,
                     dist_from_start: Dict[str, float],
                     start: str,
                     target_half_cost: float) -> str:
    """
    출발지 기준 비용이 target_half_cost 이하인 노드 중
    POI 보너스를 감안해 가중치 샘플링으로 경유지를 선택합니다.
    같은 경로가 반복되지 않도록 랜덤성을 추가합니다.
    """
    candidates = [
        nid for nid, d in dist_from_start.items()
        if 0 < d <= target_half_cost and nid != start
    ]

    if not candidates:
        raise WalkRouteFinderError(
            "목표 거리에 맞는 경유지를 찾을 수 없습니다. "
            "산책 시간을 늘리거나 지도 데이터를 확인하세요."
        )

    # 가중치: 비용이 낮을수록 선호 + 랜덤 요소
    weights = []
    for nid in candidates:
        w = 1.0 / (dist_from_start[nid] + 1.0)
        w *= random.uniform(0.8, 1.2)   # 랜덤 가중 샘플링
        weights.append(w)

    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen


def find_walk_route(map_filepath: str,
                    start_node: str,
                    walk_minutes: int,
                    seed: Optional[int] = None) -> RouteResult:
    """
    메인 함수: 산책 경로를 계산해 RouteResult를 반환합니다.

    Args:
        map_filepath : 구글맵 JSON 파일 경로
        start_node   : 출발 노드 ID (GPS 위치와 가장 가까운 노드)
        walk_minutes : 사용자가 선택한 산책 시간 (분)
        seed         : 재현 가능한 테스트용 시드 (None이면 매번 다른 결과)

    Returns:
        RouteResult  : 경로 노드 목록, 거리, 시간, 주요 포인트 등
    """
    if seed is not None:
        random.seed(seed)

    # 맵 로드
    graph = MapGraph()
    graph.load_json(map_filepath)

    if start_node not in graph.nodes:
        raise WalkRouteFinderError(f"출발 노드 '{start_node}'가 맵에 없습니다.")

    # 목표 거리 (m) 계산
    target_distance_m = WALK_SPEED_M_PER_MIN * walk_minutes

    # [수정 1] 노이즈를 다익스트라 실행 전 1회만 적용
    noisy_costs = build_noisy_costs(graph)

    # ── Step 1: 출발지 → 전체 다익스트라
    dist_from_start, prev_from_start, via_from_start = _dijkstra(
        graph, start_node, noisy_costs
    )

    # ── Step 2: 경유지 선택
    # [수정 2] 단순 1.0 대신 그래프 실제 평균 비용/m 을 사용해 단위 불일치 해소
    avg_cost_per_m = graph.compute_avg_cost_per_m()
    target_half_cost = (target_distance_m / 2) * avg_cost_per_m

    waypoint = _select_waypoint(graph, dist_from_start, start_node, target_half_cost)

    # ── Step 3: 갈 때 경로 확정
    path_out, edges_out = _reconstruct_path(
        prev_from_start, via_from_start, start_node, waypoint
    )

    # ── Step 4: 돌아오는 경로 — 다른 노이즈 + 기왕에 지나온 간선 패널티 적용
    # [수정 3] 왕복이 동일 경로가 되는 문제 해소
    return_costs = _build_return_costs(build_noisy_costs(graph), edges_out)
    dist_from_wp, prev_from_wp, via_from_wp = _dijkstra(
        graph, waypoint, return_costs
    )

    path_back, edges_back = _reconstruct_path(
        prev_from_wp, via_from_wp, waypoint, start_node
    )

    # 중복 제거 (경유지 노드가 두 번 포함되지 않도록)
    full_path_nodes = path_out + path_back[1:]
    full_path_edges = edges_out + edges_back

    # ── Step 5: 실제 거리 / 시간 / 비용 계산
    total_distance_m = sum(
        graph.edge_map[eid].distance_m
        for eid in full_path_edges
        if eid in graph.edge_map
    )
    estimated_time = total_distance_m / WALK_SPEED_M_PER_MIN
    total_cost = dist_from_start[waypoint] + dist_from_wp[start_node]

    # ── Step 6: 하이라이트 / 경고 / 참고사항 수집
    # [수정 4] 위험(construction) → warnings, 경미(alley) → notes 로 분리
    highlights: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    seen_highlights = set()

    for eid in full_path_edges:
        if eid not in graph.edge_map:
            continue
        e = graph.edge_map[eid]

        if e.road_type in ("boulevard", "park") and e.name not in seen_highlights:
            highlights.append(f"✅ {e.name} ({e.road_type})")
            seen_highlights.add(e.name)

        if "pet_store" in e.nearby_pois and "반려동물 용품점" not in seen_highlights:
            highlights.append("🐾 반려동물 용품점 근처를 지납니다")
            seen_highlights.add("반려동물 용품점")

        if e.road_type == "construction":
            warnings.append(f"⚠️ {e.name}: 공사 구간 포함 ({e.notes})")

        if e.road_type == "alley":
            notes.append(f"ℹ️ {e.name}: 좁은 골목길이 포함됩니다")

    node_names = [graph.nodes[nid].name for nid in full_path_nodes]

    return RouteResult(
        node_ids=full_path_nodes,
        node_names=node_names,
        edge_ids=full_path_edges,
        total_distance_m=round(total_distance_m, 1),
        estimated_time_min=round(estimated_time, 1),
        total_cost=round(total_cost, 2),
        highlights=highlights,
        warnings=warnings,
        notes=notes,
    )


# ─────────────────────────────────────────────
# 6. GPS → 최근접 노드 변환 (Android 연동용)
# ─────────────────────────────────────────────

def find_nearest_node(graph: MapGraph, lat: float, lng: float) -> str:
    """
    현재 GPS 좌표와 가장 가까운 노드 ID를 반환합니다.
    Android에서 10초마다 위치 업데이트 시 호출하세요.
    """
    def haversine(lat1, lng1, lat2, lng2) -> float:
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    nearest = min(
        graph.nodes.values(),
        key=lambda n: haversine(lat, lng, n.lat, n.lng)
    )
    return nearest.id


# ─────────────────────────────────────────────
# 7. 결과 출력 헬퍼
# ─────────────────────────────────────────────

def print_route_result(result: RouteResult):
    print("=" * 55)
    print("🐕 반려동물 산책 경로 추천 결과")
    print("=" * 55)
    print(f"📍 경로 노드 수  : {len(result.node_ids)}개")
    print(f"📏 총 거리      : {result.total_distance_m:.0f} m  ({result.total_distance_m/1000:.2f} km)")
    print(f"⏱  예상 시간    : {result.estimated_time_min:.0f} 분")
    print(f"💡 알고리즘 비용 : {result.total_cost:.1f}  (낮을수록 쾌적한 경로)")
    print()
    print("─ 경로 순서 ─")
    for i, name in enumerate(result.node_names):
        arrow = " → " if i < len(result.node_names) - 1 else " (귀환)"
        print(f"  [{i+1:02d}] {name}{arrow}")
    print()
    if result.highlights:
        print("─ 주요 포인트 ─")
        for h in result.highlights:
            print(f"  {h}")
        print()
    if result.warnings:
        print("─ 주의사항 ─")
        for w in result.warnings:
            print(f"  {w}")
        print()
    if result.notes:
        print("─ 참고사항 ─")
        for n in result.notes:
            print(f"  {n}")
        print()
    print("=" * 55)


# ─────────────────────────────────────────────
# 8. 실행 예시
# ─────────────────────────────────────────────

if __name__ == "__main__":
    MAP_FILE = "map_data.json"
    START_NODE = "N00"
    WALK_MINUTES = 30

    print(f"\n산책 시간: {WALK_MINUTES}분  |  목표 거리: {WALK_SPEED_M_PER_MIN * WALK_MINUTES:.0f}m\n")

    for trial in range(1, 4):
        print(f"\n{'━'*20} 시도 {trial} {'━'*20}")
        try:
            result = find_walk_route(MAP_FILE, START_NODE, WALK_MINUTES)
            print_route_result(result)
        except WalkRouteFinderError as e:
            print(f"❌ 경로 탐색 실패: {e}")

    print("\n[GPS → 노드 변환 예시]")
    graph = MapGraph()
    graph.load_json(MAP_FILE)
    gps_lat, gps_lng = 37.5671, 126.9781
    nearest = find_nearest_node(graph, gps_lat, gps_lng)
    print(f"  GPS ({gps_lat}, {gps_lng}) → 최근접 노드: {nearest} ({graph.nodes[nearest].name})")