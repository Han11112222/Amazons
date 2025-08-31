from __future__ import annotations
import time, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import streamlit as st

# =============== 기본 설정 ===============
st.set_page_config(page_title="Amazons (1P vs CPU)", layout="wide")
EMO_HUM = "🔵"   # 플레이어 말
EMO_CPU = "🟡"   # 컴퓨터 말
EMO_BLK = "⬛"   # 바위(막힘)
EMO_EMPTY = "·"

SIZE = 10  # 10x10 보드
EMPTY, HUM, CPU, BLOCK = 0, 1, 2, 3
DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

@dataclass
class Move:
    fr: Tuple[int,int]
    to: Tuple[int,int]
    shot: Tuple[int,int]

Board = List[List[int]]

# =============== 보드/규칙 유틸 ===============
def in_bounds(r:int,c:int)->bool:
    return 0 <= r < SIZE and 0 <= c < SIZE

def clone(board:Board)->Board:
    return [row[:] for row in board]

def iter_ray(board:Board, r:int,c:int, dr:int,dc:int):
    nr, nc = r+dr, c+dc
    while in_bounds(nr,nc) and board[nr][nc]==EMPTY:
        yield (nr,nc)
        nr += dr; nc += dc

def piece_positions(board:Board, side:int)->List[Tuple[int,int]]:
    token = HUM if side==HUM else CPU
    res=[]
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c]==token:
                res.append((r,c))
    return res

def legal_dests_from(board:Board, r:int,c:int)->List[Tuple[int,int]]:
    dests=[]
    for dr,dc in DIRS:
        for (nr,nc) in iter_ray(board,r,c,dr,dc):
            dests.append((nr,nc))
    return dests

def legal_shots_from(board:Board, r:int,c:int)->List[Tuple[int,int]]:
    shots=[]
    for dr,dc in DIRS:
        for (nr,nc) in iter_ray(board,r,c,dr,dc):
            shots.append((nr,nc))
    return shots

def apply_move(board:Board, mv:Move, side:int)->Board:
    nb = clone(board)
    fr, to, shot = mv.fr, mv.to, mv.shot
    r1,c1 = fr; r2,c2 = to; rs,cs = shot
    nb[r1][c1] = EMPTY
    nb[r2][c2] = side
    nb[rs][cs] = BLOCK
    return nb

def has_any_move(board:Board, side:int)->bool:
    for r,c in piece_positions(board, side):
        if legal_dests_from(board,r,c):
            return True
    return False

# =============== 평가/AI ===============
def mobility(board:Board, side:int)->int:
    # 말이 이동 가능한 목적지 수(화살 제외)
    m=0
    for r,c in piece_positions(board, side):
        m += len(legal_dests_from(board,r,c))
    return m

def liberties(board:Board, side:int)->int:
    # 말 주변 8방의 빈칸 수
    s=0
    for r,c in piece_positions(board, side):
        for dr,dc in DIRS:
            nr,nc=r+dr,c+dc
            if in_bounds(nr,nc) and board[nr][nc]==EMPTY:
                s+=1
    return s

def center_score(board:Board, side:int)->int:
    # 중앙 근접도(가까울수록 좋음)
    cx, cy = (SIZE-1)/2, (SIZE-1)/2
    total=0
    for r,c in piece_positions(board, side):
        total -= int(abs(r-cx)+abs(c-cy))
    return total

def evaluate(board:Board)->int:
    # + 점수면 CPU 유리, - 면 플레이어 유리
    cpu_m = mobility(board, CPU)
    hum_m = mobility(board, HUM)
    cpu_l = liberties(board, CPU)
    hum_l = liberties(board, HUM)
    cpu_c = center_score(board, CPU)
    hum_c = center_score(board, HUM)
    return 10*(cpu_m-hum_m) + 2*(cpu_l-hum_l) + (cpu_c-hum_c)

def generate_moves_limited(board:Board, side:int, k_dest:int, k_shot:int, cap_total:int)->List[Move]:
    """분기 제한형 생성: 목적지 상위 k, 샷 상위 k, 전체 cap_total"""
    moves: List[Move] = []
    for r,c in piece_positions(board, side):
        dests = legal_dests_from(board, r, c)
        scored_d=[]
        for (tr,tc) in dests:
            # 이동만 적용하여 임시 평가(빠른 휴리스틱)
            tmp = clone(board)
            tmp[r][c]=EMPTY; tmp[tr][tc]=side
            # 상대 이동성 감소/자신 이동성 증가를 선호
            sc = (mobility(tmp, side) - mobility(tmp, HUM if side==CPU else CPU))
            # 중앙 선호
            sc += - (abs(tr-(SIZE-1)/2)+abs(tc-(SIZE-1)/2))/5
            scored_d.append(((tr,tc), sc))
        scored_d.sort(key=lambda x:x[1], reverse=True)
        for (tr,tc), _ in scored_d[:k_dest]:
            tmp = clone(board)
            tmp[r][c]=EMPTY; tmp[tr][tc]=side
            shots = legal_shots_from(tmp, tr, tc)
            scored_s=[]
            for (sr,sc) in shots:
                tmp2 = clone(tmp); tmp2[sr][sc]=BLOCK
                s = (mobility(tmp2, side) - mobility(tmp2, HUM if side==CPU else CPU))
                scored_s.append(((sr,sc), s))
            scored_s.sort(key=lambda x:x[1], reverse=True)
            for (sr,sc), _ in scored_s[:k_shot]:
                moves.append(Move((r,c),(tr,tc),(sr,sc)))
                if len(moves)>=cap_total: return moves
    return moves

def search(board:Board, depth:int, alpha:int, beta:int, side:int,
           params:Dict[str,int])->int:
    # 종료
    if depth==0 or not has_any_move(board, side):
        # 움직일 수 없는 쪽이 지금 차례면 패배 → 상대에게 유리(부호)
        if not has_any_move(board, side):
            return 10_000 if side==HUM else -10_000  # side 가 HUM 차례인데 HUM이 못움직이면 CPU 유리(큰 +)
        return evaluate(board)

    k_dest = params["k_dest_d{}".format(depth)]
    k_shot = params["k_shot_d{}".format(depth)]
    cap = params["cap_d{}".format(depth)]

    moves = generate_moves_limited(board, side, k_dest, k_shot, cap)
    if not moves:
        return 10_000 if side==HUM else -10_000

    if side==CPU:
        best=-1_000_000
        for mv in moves:
            nb = apply_move(board, mv, CPU)
            val = search(nb, depth-1, alpha, beta, HUM, params)
            if val>best: best=val
            alpha = max(alpha, val)
            if beta<=alpha: break
        return best
    else:
        best=1_000_000
        for mv in moves:
            nb = apply_move(board, mv, HUM)
            val = search(nb, depth-1, alpha, beta, CPU, params)
            if val<best: best=val
            beta = min(beta, val)
            if beta<=alpha: break
        return best

def ai_move(board:Board, difficulty:int)->Optional[Move]:
    # 난이도에 따른 탐색 파라미터(깊이/분기)
    # d1~3: 1수만, d4~6: 2수, d7~10: 3수(제한)
    if difficulty<=3:
        depth=1
        params = dict(k_dest_d1=6+difficulty*4, k_shot_d1=4+difficulty*2, cap_d1=40+difficulty*20)
    elif difficulty<=6:
        depth=2
        params = dict(
            k_dest_d2=6+2*(difficulty-3), k_shot_d2=4+1*(difficulty-3), cap_d2=40+10*(difficulty-3),
            k_dest_d1=10, k_shot_d1=8, cap_d1=80
        )
    else:
        depth=3
        scale = difficulty-6
        params = dict(
            k_dest_d3=4+scale,   k_shot_d3=3+scale//2, cap_d3=18+4*scale,
            k_dest_d2=8+scale,   k_shot_d2=6+scale//2, cap_d2=40+8*scale,
            k_dest_d1=10,        k_shot_d1=8,         cap_d1=80
        )

    # 루트 분기 생성
    root_moves = generate_moves_limited(board, CPU,
                                        params.get("k_dest_d{}".format(depth), 8),
                                        params.get("k_shot_d{}".format(depth), 6),
                                        params.get("cap_d{}".format(depth), 50))
    if not root_moves:
        return None

    best_mv=None; best_val=-1_000_000
    for mv in root_moves:
        nb = apply_move(board, mv, CPU)
        val = search(nb, depth-1, -1_000_000, 1_000_000, HUM, params)
        if val>best_val:
            best_val=val; best_mv=mv
    return best_mv

# =============== 초기 배치(표준 10×10) ===============
def initial_board()->Board:
    b = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    # White(플레이어) d1,g1,a4,j4  -> 0-index: (9,3),(9,6),(6,0),(6,9)
    b[9][3]=HUM; b[9][6]=HUM; b[6][0]=HUM; b[6][9]=HUM
    # Black(컴퓨터) a7,j7,d10,g10 -> (3,0),(3,9),(0,3),(0,6)
    b[3][0]=CPU; b[3][9]=CPU; b[0][3]=CPU; b[0][6]=CPU
    return b

# =============== 세션 상태 ===============
def reset_game():
    st.session_state.board = initial_board()
    st.session_state.turn = HUM   # 플레이어 선공
    st.session_state.phase = "select"  # select -> move -> shoot
    st.session_state.sel_from = None
    st.session_state.sel_to = None
    st.session_state.legal = set()
    st.session_state.difficulty = st.session_state.get("difficulty", 5)
    st.session_state.message = ""

if "board" not in st.session_state:
    reset_game()

# =============== 상단 컨트롤 ===============
left, right = st.columns([1,1])
with left:
    st.title("목우회 Amazons (1P vs CPU)")
    st.caption("규칙: 말 한 개를 퀸처럼 이동 → 그 자리에서 바위를 퀸처럼 발사해 빈칸을 막기. 상대가 더 이상 이동 못하면 승리.")
with right:
    diff = st.slider("난이도 (1 쉬움 ··· 10 어려움)", 1, 10, st.session_state.get("difficulty",5))
    st.session_state.difficulty = diff
    c1,c2,c3 = st.columns(3)
    if c1.button("새 게임", use_container_width=True): reset_game()
    if c2.button("되돌리기(1수)", use_container_width=True):
        hist: List[Board] = st.session_state.get("hist", [])
        if hist: st.session_state.board = hist.pop()
    st.session_state.setdefault("hist", [])

board: Board = st.session_state.board

# =============== 보드 렌더/입력 ===============
def render_board():
    st.markdown("#### 보드")
    # 전설(legend)
    st.caption(f"{EMO_HUM}=플레이어 {EMO_CPU}=컴퓨터 {EMO_BLK}=바위")
    for r in range(SIZE):
        cols = st.columns(SIZE)
        for c in range(SIZE):
            cell = board[r][c]
            label = EMO_EMPTY
            if cell==HUM: label = EMO_HUM
            elif cell==CPU: label = EMO_CPU
            elif cell==BLOCK: label = EMO_BLK

            key = f"b_{r}_{c}"
            clickable=False
            # 클릭 가능 조건
            if st.session_state.turn==HUM:
                if st.session_state.phase=="select" and cell==HUM:
                    clickable=True
                elif st.session_state.phase=="move" and (r,c) in st.session_state.legal:
                    clickable=True
                elif st.session_state.phase=="shoot" and (r,c) in st.session_state.legal:
                    clickable=True

            if clickable:
                if cols[c].button(label, key=key):
                    handle_click(r,c)
            else:
                cols[c].markdown(f"<div style='text-align:center;font-size:22px'>{label}</div>", unsafe_allow_html=True)

def handle_click(r:int,c:int):
    turn = st.session_state.turn
    phase = st.session_state.phase
    if turn!=HUM: return

    if phase=="select":
        if board[r][c]==HUM:
            st.session_state.sel_from = (r,c)
            st.session_state.phase = "move"
            st.session_state.legal = set(legal_dests_from(board,r,c))
            st.rerun()

    elif phase=="move":
        if (r,c) in st.session_state.legal:
            # 이동 반영(임시)
            fr = st.session_state.sel_from
            nb = clone(board)
            nb[fr[0]][fr[1]] = EMPTY
            nb[r][c] = HUM
            st.session_state.preview = nb
            st.session_state.sel_to = (r,c)
            st.session_state.phase = "shoot"
            st.session_state.legal = set(legal_shots_from(nb,r,c))
            st.session_state.board = nb
            st.rerun()

    elif phase=="shoot":
        if (r,c) in st.session_state.legal:
            # 최종 확정
            st.session_state.board[r][c] = BLOCK
            st.session_state.hist.append(clone(board))  # undo용
            st.session_state.phase = "select"
            st.session_state.turn = CPU
            st.session_state.sel_from = None
            st.session_state.sel_to = None
            st.session_state.legal = set()
            st.rerun()

render_board()

# =============== 차례 처리 ===============
def announce(msg:str, success=True):
    color = "#16a34a" if success else "#dc2626"
    st.markdown(f"<div style='padding:8px;border-radius:8px;background:{'#ecfdf5' if success else '#fef2f2'};color:{color}'>{msg}</div>", unsafe_allow_html=True)

if st.session_state.turn==HUM:
    if not has_any_move(board,HUM):
        announce("컴퓨터 승리! (플레이어가 움직일 곳이 없음)", success=False)
else:
    if not has_any_move(board,CPU):
        announce("플레이어 승리! (컴퓨터가 움직일 곳이 없음)")
    else:
        with st.spinner("컴퓨터 생각중..."):
            mv = ai_move(board, st.session_state.difficulty)
            if mv is None:
                announce("플레이어 승리! (컴퓨터가 움직일 곳이 없음)")
            else:
                st.session_state.hist.append(clone(board))
                st.session_state.board = apply_move(board, mv, CPU)
                st.session_state.turn = HUM
                st.session_state.phase = "select"
                st.session_state.sel_from = None
                st.session_state.sel_to = None
                st.session_state.legal = set()
        st.rerun()
